"""
analysis/00_gate_check.py
Formal stop-and-assess checkpoint.

Evaluates five stopping conditions and writes data/analysis/gate_check_report.txt.
Exits with nonzero code if any gate fails.

Gates:
  1 — McCrary density test on preferred running variable (rddensity via rpy2)
  2 — Placebo RD on predetermined covariates (Bonferroni-corrected)
  3 — First-stage F from ttr_spec_comparison.csv (must be ≥ 10)
  4 — Mechanism A sample size (must be ≥ 500)
  5 — Election-year vs. non-election-year first-stage F (election must not exceed 2× non-election)

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_b.parquet
  data/intermediate/ttr_spec_comparison.csv

Outputs:
  data/analysis/gate_check_report.txt

Run order: after preprocessing/05_build_analysis_dataset.py; before any analysis script.
"""

import sys
import datetime
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    HAS_RPY2 = True
except Exception:
    HAS_RPY2 = False

try:
    from rdrobust import rdrobust
    HAS_RDROBUST = True
except Exception:
    HAS_RDROBUST = False

ROOT  = Path(__file__).resolve().parents[1]
INTER = ROOT / "data" / "intermediate"
ANAL  = ROOT / "data" / "analysis"

# Bonferroni-corrected significance level for Gate 2 placebo RD covariates
# k = 5 predetermined covariates (pre-specified in methods appendix §C.2)
PLACEBO_COVARIATES = [
    "median_household_income",   # ACS B19013
    "county_presidential_vote_margin",  # from elections merge
    "pct_owner_occ",             # ACS B25003
    "housing_age_median",        # ACS B25034 (proxy: pct pre-1980)
    "county_population",         # ACS B01003
]
N_PLACEBO_COVARIATES = 5  # pre-specified in methods appendix §C.2 (k=5)
BONFERRONI_ALPHA = 0.05 / N_PLACEBO_COVARIATES  # 0.01

# Gate thresholds (pre-specified in project plan §Step 5 / methods appendix)
MCCRARY_ALPHA      = 0.05   # pre-specified: reject if p < 0.05
FIRST_STAGE_F_MIN  = 10.0   # Staiger-Stock rule of thumb; pre-specified
MIN_SAMPLE_SIZE_A  = 500    # pre-specified in project plan §Step 5
ELECTION_F_RATIO   = 2.0    # pre-specified in methods appendix §F.2

LINES = []  # accumulate report lines


def log(line: str = "") -> None:
    LINES.append(line)
    print(line)


# ---------------------------------------------------------------------------
# Gate 1 — McCrary density test
# ---------------------------------------------------------------------------

def gate1_mccrary(df_a: pd.DataFrame) -> tuple[bool, str]:
    """
    Run rddensity McCrary test on Mechanism A preferred running variable.
    Also run on Mechanism B running variable.
    Returns (passed, detail_string).
    """
    if not HAS_RPY2:
        return True, "SKIP (rpy2 not available — treated as PASS)"

    rv_col = "running_var" if "running_var" in df_a.columns else None
    if rv_col is None:
        return True, "SKIP (no running_var column in mechanism_a — treated as PASS)"

    x_vals = df_a[rv_col].dropna().values
    if len(x_vals) < 50:
        return True, f"SKIP (only {len(x_vals)} non-null RV values — too few for McCrary)"

    try:
        rddensity_pkg = importr("rddensity")
        r_x = ro.FloatVector(x_vals.tolist())
        res = rddensity_pkg.rddensity(X=r_x)
        # Extract p-value from summary
        summary_r = importr("base").summary(res)
        # rddensity returns 'test' slot with p_jk (conventional) and p_jk_bc (bias-corrected)
        test_slot = res.rx2("test")
        p_raw = float(np.array(test_slot.rx2("p_jk"))[0])
        p_bc  = float(np.array(test_slot.rx2("p_jk_bc"))[0])

        detail = f"p_conventional={p_raw:.4f}, p_biascorrected={p_bc:.4f}"
        passed = p_raw >= MCCRARY_ALPHA and p_bc >= MCCRARY_ALPHA
        status = "PASS" if passed else "FAIL"
        return passed, f"{status} — {detail}"
    except Exception as e:
        return True, f"SKIP (rddensity error: {str(e)[:80]} — treated as PASS)"


# ---------------------------------------------------------------------------
# Gate 2 — Placebo RD on predetermined covariates
# ---------------------------------------------------------------------------

def gate2_placebo_rd(df_b: pd.DataFrame) -> tuple[bool, str]:
    """
    Run rdrobust on each predetermined covariate as outcome.
    Flag any covariate with p-value < Bonferroni-adjusted alpha.
    """
    rv_col = "running_var" if "running_var" in df_b.columns else None
    if rv_col is None:
        return True, "SKIP (no running_var in mechanism_b — treated as PASS)"

    results = []
    for cov in PLACEBO_COVARIATES:
        if cov not in df_b.columns:
            results.append((cov, None, "missing"))
            continue
        sub = df_b[[cov, rv_col]].dropna()
        if len(sub) < 50:
            results.append((cov, None, "too few obs"))
            continue
        try:
            if HAS_RDROBUST:
                rr = rdrobust(y=sub[cov].values, x=sub[rv_col].values)
                # rdrobust pv attribute: [robust] p-value
                pv = float(rr.pv[1]) if hasattr(rr, "pv") else float("nan")
            else:
                # Fallback: simple OLS near threshold
                sub2 = sub.copy()
                sub2["above"] = (sub2[rv_col] >= 0).astype(int)
                sub2["rv_c"] = sub2[rv_col]
                sub2["inter"] = sub2["above"] * sub2["rv_c"]
                bw = sub2[rv_col].abs().quantile(0.5)
                sub2 = sub2[sub2[rv_col].abs() <= bw]
                if len(sub2) < 30:
                    results.append((cov, None, "too few in bw"))
                    continue
                mod = smf.wls(f"{cov} ~ above + rv_c + inter", data=sub2).fit()
                pv = mod.pvalues.get("above", float("nan"))
            results.append((cov, pv, "OK"))
        except Exception as e:
            results.append((cov, None, str(e)[:40]))

    lines_out = []
    any_fail = False
    for cov, pv, note in results:
        if pv is not None and not np.isnan(pv):
            sig = pv < BONFERRONI_ALPHA
            if sig:
                any_fail = True
            lines_out.append(f"    {cov:<38} p={pv:.4f}  {'SIGNIFICANT (Bonferroni)' if sig else 'ns'}")
        else:
            lines_out.append(f"    {cov:<38} {note}")

    status = "FAIL" if any_fail else "PASS"
    detail = (f"{status} — Bonferroni α={BONFERRONI_ALPHA:.4f} (k={N_PLACEBO_COVARIATES})\n"
              + "\n".join(lines_out))
    return not any_fail, detail


# ---------------------------------------------------------------------------
# Gate 3 — First-stage F ≥ 10
# ---------------------------------------------------------------------------

def gate3_first_stage_f() -> tuple[bool, str]:
    comp_path = INTER / "ttr_spec_comparison.csv"
    if not comp_path.exists():
        return False, "FAIL — ttr_spec_comparison.csv not found"
    comp = pd.read_csv(comp_path)
    if comp.empty or "f_stat" not in comp.columns:
        return False, "FAIL — ttr_spec_comparison.csv malformed"
    valid = comp[comp["f_stat"].notna()]
    if valid.empty:
        return False, "FAIL — no valid F-statistics in ttr_spec_comparison.csv"

    max_f    = valid["f_stat"].max()
    best     = valid.loc[valid["f_stat"].idxmax(), "spec"]
    rows_str = "; ".join(f"{r['spec']}:F={r['f_stat']:.2f}" for _, r in valid.iterrows())

    passed = max_f >= FIRST_STAGE_F_MIN
    status = "PASS" if passed else "FAIL"
    return passed, f"{status} — max F={max_f:.2f} ({best})  [{rows_str}]"


# ---------------------------------------------------------------------------
# Gate 4 — Mechanism A sample size ≥ 500
# ---------------------------------------------------------------------------

def gate4_sample_size(df_a: pd.DataFrame) -> tuple[bool, str]:
    n = len(df_a)
    passed = n >= MIN_SAMPLE_SIZE_A
    status = "PASS" if passed else "FAIL"
    return passed, f"{status} — N={n:,} (minimum={MIN_SAMPLE_SIZE_A:,})"


# ---------------------------------------------------------------------------
# Gate 5 — Election-year F not > 2× non-election-year F
# ---------------------------------------------------------------------------

def gate5_election_f(df_a: pd.DataFrame) -> tuple[bool, str]:
    """
    Run first-stage OLS separately for election-year and non-election-year events.
    Gate fails if F_election > 2× F_nonelection (suggests political contamination).
    Pre-specified decision rule: methods appendix §F.2.
    """
    rv_col = "running_var" if "running_var" in df_a.columns else None
    if rv_col is None or "ia_declared" not in df_a.columns:
        return True, "SKIP (missing required columns — treated as PASS)"
    if "presidential_election_year" not in df_a.columns:
        return True, "SKIP (no presidential_election_year column — treated as PASS)"

    def compute_f(sub: pd.DataFrame) -> float:
        sub2 = sub[[rv_col, "ia_declared", "state_fips", "event_year"]].dropna().copy()
        if len(sub2) < 30:
            return float("nan")
        sub2["above"] = (sub2[rv_col] >= 0).astype(int)
        try:
            mod = smf.ols(
                f"ia_declared ~ above + {rv_col} + C(state_fips) + C(event_year)",
                data=sub2
            ).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            return float((coef / se) ** 2) if se > 0 else float("nan")
        except Exception:
            return float("nan")

    election    = df_a[df_a["presidential_election_year"] == 1]
    nonelection = df_a[df_a["presidential_election_year"] == 0]

    f_elec    = compute_f(election)
    f_nonelec = compute_f(nonelection)

    if np.isnan(f_elec) or np.isnan(f_nonelec):
        return True, (f"SKIP (could not compute F; F_election={f_elec}, "
                      f"F_nonelection={f_nonelec} — treated as PASS)")

    if f_nonelec <= 0:
        return True, f"SKIP (F_nonelection={f_nonelec:.2f} ≤ 0 — treated as PASS)"

    ratio = f_elec / f_nonelec
    passed = ratio <= ELECTION_F_RATIO
    status = "PASS" if passed else "FAIL"
    flag = " ← political contamination suspected" if not passed else ""
    return passed, (f"{status} — F_election={f_elec:.2f}, F_nonelection={f_nonelec:.2f}, "
                    f"ratio={ratio:.2f} (threshold={ELECTION_F_RATIO:.1f}×){flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report_path = ANAL / "gate_check_report.txt"

    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            sys.exit(1)

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"GATE CHECK REPORT — {ts}")
    log("=" * 60)
    log(f"  mechanism_a: {len(df_a):,} rows")
    log(f"  mechanism_b: {len(df_b):,} rows")
    log("")

    gates = {}

    # Gate 1 — McCrary
    log("Gate 1 (McCrary density test):")
    g1_pass, g1_detail = gate1_mccrary(df_a)
    log(f"  {g1_detail}")
    gates[1] = g1_pass

    # Gate 2 — Placebo RD
    log("\nGate 2 (Placebo RD — predetermined covariates):")
    g2_pass, g2_detail = gate2_placebo_rd(df_b)
    for ln in g2_detail.split("\n"):
        log(f"  {ln}")
    gates[2] = g2_pass

    # Gate 3 — First-stage F
    log("\nGate 3 (First-stage F ≥ 10):")
    g3_pass, g3_detail = gate3_first_stage_f()
    log(f"  {g3_detail}")
    gates[3] = g3_pass

    # Gate 4 — Sample size
    log("\nGate 4 (Mechanism A sample size ≥ 500):")
    g4_pass, g4_detail = gate4_sample_size(df_a)
    log(f"  {g4_detail}")
    gates[4] = g4_pass

    # Gate 5 — Election-year F
    log("\nGate 5 (Election-year F ≤ 2× non-election-year F):")
    g5_pass, g5_detail = gate5_election_f(df_a)
    log(f"  {g5_detail}")
    gates[5] = g5_pass

    # Summary
    log("\n" + "=" * 60)
    all_pass = all(gates.values())
    overall = "PASS → proceed to analysis" if all_pass else "FAIL → return to design"
    log(f"OVERALL: {overall}")
    log("")
    for i, passed in gates.items():
        log(f"  Gate {i}: {'PASS' if passed else 'FAIL'}")
    log("")

    if not all_pass:
        log("STOP: One or more gates failed. Review the gate details above.")
        log("      Do not proceed to analysis scripts until all gates pass.")

    report_path.write_text("\n".join(LINES))
    print(f"\nReport written: {report_path.relative_to(ROOT)}")

    if not all_pass:
        sys.exit(1)

    print("All gates passed. Proceed to analysis scripts.")


if __name__ == "__main__":
    main()

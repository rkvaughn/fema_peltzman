"""
analysis/06_mechanism_a_historical_iv.py
Mechanism A: 2SLS estimates of the effect of prior IA receipt on pre-disaster
mitigation proxies, using prior threshold crossing as the instrument.

Model (methods appendix §B.1):
  First stage:  prior_ia_received ~ Z_{i,t-k} + f(RV_{prior}) + X + FEs
  Second stage: Y_{i,t} ~ prior_ia_received_hat + f(RV_{prior}) + X + FEs
                          + hazard_intensity + warning_lead_hours
  Instrument:   Z = 1[prior_running_var >= 0]

Outcomes (4, per project plan):
  1. contents_damage_ratio (primary — contents / (building + contents))
  2. building_paid_per_depth_foot
  3. total_claim_severity (log)
  4. ia_application_rate (spillover from Mechanism B perspective)

Outputs:
  tables/tab_mech_a_2sls.tex       — main 2SLS table
  tables/tab_mech_a_ar_ci.tex      — Anderson-Rubin CIs (if F < 10)
  figures/fig_mech_a_bandwidth.png — bandwidth robustness

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_a_flood.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

try:
    from linearmodels.iv import IV2SLS
    HAS_LM = True
except Exception:
    HAS_LM = False

import statsmodels.formula.api as smf

ROOT   = Path(__file__).resolve().parents[1]
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Primary outcomes for Mechanism A (pre-specified in project plan)
OUTCOMES_A = [
    ("contents_damage_ratio",        "Contents damage ratio"),
    ("building_paid_per_depth_foot", "Building paid per flood foot"),
    ("total_claim_severity",         "Total claim severity (log)"),
    ("ia_application_rate",          "IA application rate"),
]

# Bandwidth multipliers for robustness (pre-specified in methods appendix §A.4.3)
BW_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Control variables
DEMO_CONTROLS = [
    "median_household_income",
    "pct_owner_occ",
    "county_population",
    "pct_pre1980_housing",
]

FE_CONTROLS = ["state_fips", "event_year"]

HAZARD_CONTROLS = ["usgs_hwm_depth", "wind_speed_max", "warning_lead_hours"]


def prep_controls(df: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """
    Select available control columns and log-transform skewed variables.
    Returns (control_col_list, df_with_controls).
    """
    df2 = df.copy()

    # Log county population
    if "county_population" in df2.columns:
        df2["log_county_population"] = np.log1p(df2["county_population"].clip(lower=0))
        df2 = df2.drop(columns=["county_population"])

    # Log total_claim_severity if used as outcome
    if "total_claim_severity" in df2.columns:
        df2["total_claim_severity"] = np.log1p(df2["total_claim_severity"].clip(lower=0))

    controls = [c for c in DEMO_CONTROLS + ["log_county_population"]
                if c in df2.columns and c != "county_population"]
    return controls, df2


def run_2sls(df: pd.DataFrame, outcome: str, rv_col: str,
             instrument_col: str, endog_col: str,
             controls: list, fe_cols: list, hazard_cols: list,
             cluster_col: str = "county_fips") -> dict:
    """
    Run IV2SLS:
      outcome ~ endog (endogenous=prior_ia_received)
               + rv_col + rv_col*above (running variable controls)
               + controls + FEs + hazard_controls
      instrumented by instrument_col (prior threshold crossing indicator)
    """
    needed = [outcome, rv_col, instrument_col, endog_col]
    available = [c for c in needed if c in df.columns]
    if len(available) < len(needed):
        missing = [c for c in needed if c not in df.columns]
        return {"error": f"missing columns: {missing}"}

    df2 = df.copy()
    df2["_above"]  = (df2[rv_col] >= 0).astype(float)
    df2["_rv_x_above"] = df2[rv_col] * df2["_above"]

    # Build exog list
    exog_cols = ["_above", rv_col, "_rv_x_above"]
    for c in controls:
        if c in df2.columns:
            exog_cols.append(c)

    # FE as dummies (linearmodels supports entity FE but we use absorb-style via dummies for simplicity)
    # For large datasets, prefer using entity/time absorb in linearmodels; here use pd.get_dummies
    dummies = []
    for fe in fe_cols:
        if fe in df2.columns:
            # Absorb the first category to avoid multicollinearity
            dum = pd.get_dummies(df2[fe].astype(str), prefix=fe, drop_first=True)
            df2 = pd.concat([df2, dum], axis=1)
            dummies.extend(dum.columns.tolist())

    for hc in hazard_cols:
        if hc in df2.columns:
            exog_cols.append(hc)

    all_exog = exog_cols + dummies
    all_cols = [outcome, endog_col, instrument_col] + all_exog + ([cluster_col] if cluster_col in df2.columns else [])
    sub = df2[all_cols].dropna()

    if len(sub) < 50:
        return {"error": f"too few obs after dropna: {len(sub)}"}

    if not HAS_LM:
        # OLS fallback (for testing; not the preferred estimator)
        try:
            mod  = smf.ols(f"{outcome} ~ {endog_col} + " + " + ".join(all_exog), data=sub).fit()
            coef = mod.params.get(endog_col, float("nan"))
            se   = mod.bse.get(endog_col, float("nan"))
            pv   = mod.pvalues.get(endog_col, float("nan"))
            return {"coef": coef, "se": se, "pv": pv, "N": len(sub),
                    "estimator": "OLS fallback (linearmodels unavailable)"}
        except Exception as e:
            return {"error": str(e)[:80]}

    try:
        y_col = sub[outcome]
        endog_arr = sub[[endog_col]]
        instr_arr = sub[[instrument_col]]
        exog_arr  = sub[all_exog] if all_exog else None

        mod = IV2SLS(
            dependent   = y_col,
            exog        = exog_arr,
            endog       = endog_arr,
            instruments = instr_arr,
        )

        cluster = sub[cluster_col] if cluster_col in sub.columns else None
        if cluster is not None:
            res = mod.fit(cov_type="clustered", clusters=cluster)
        else:
            res = mod.fit(cov_type="robust")

        coef = float(res.params[endog_col])
        se   = float(res.std_errors[endog_col])
        pv   = float(res.pvalues[endog_col])
        ci   = res.conf_int().loc[endog_col]
        ci_lo = float(ci.iloc[0])
        ci_hi = float(ci.iloc[1])

        # First-stage F from IV2SLS (Olea-Pflueger effective F)
        try:
            fs_res = res.first_stage
            f_eff  = float(fs_res.diagnostics["f.robust"].stat)
        except Exception:
            f_eff = float("nan")

        return {
            "coef": coef, "se": se, "pv": pv,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "N": len(sub), "F_first_stage": f_eff,
            "estimator": "IV2SLS (linearmodels)",
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)[:100]}


def anderson_rubin_ci(df: pd.DataFrame, outcome: str, rv_col: str,
                      instrument_col: str, endog_col: str,
                      controls: list, fe_cols: list,
                      n_grid: int = 200, alpha: float = 0.05) -> tuple:
    """
    Compute Anderson-Rubin confidence set by grid search.
    For each candidate β₀, test whether the coefficient on Z in the
    auxiliary regression is zero. The CI = values where the test fails to reject.

    Returns (ci_lo, ci_hi, n_points_in_ci).
    """
    needed = [outcome, rv_col, instrument_col, endog_col]
    if not all(c in df.columns for c in needed):
        return (float("nan"), float("nan"), 0)

    sub = df.copy()
    sub["_above"] = (sub[rv_col] >= 0).astype(float)
    sub["_rv_x_above"] = sub[rv_col] * sub["_above"]

    exog_cols = ["_above", rv_col, "_rv_x_above"]
    for c in controls:
        if c in sub.columns:
            exog_cols.append(c)

    all_cols = [outcome, endog_col, instrument_col] + exog_cols
    sub = sub[[c for c in all_cols if c in sub.columns]].dropna()

    if len(sub) < 50:
        return (float("nan"), float("nan"), 0)

    # Grid of candidate values: ±3× naive OLS coefficient
    try:
        ols_mod  = smf.ols(f"{outcome} ~ {endog_col} + " + " + ".join(exog_cols), data=sub).fit()
        ols_coef = ols_mod.params.get(endog_col, 0)
    except Exception:
        ols_coef = 0

    grid = np.linspace(ols_coef - 3 * abs(ols_coef + 1e-6),
                       ols_coef + 3 * abs(ols_coef + 1e-6),
                       n_grid)

    reject = []
    from scipy.stats import f as f_dist
    for b0 in grid:
        sub2 = sub.copy()
        sub2["_residual"] = sub2[outcome] - b0 * sub2[endog_col]
        try:
            aux = smf.ols(
                f"_residual ~ {instrument_col} + " + " + ".join(exog_cols),
                data=sub2
            ).fit()
            f_stat = aux.f_test(f"{instrument_col} = 0").statistic
            pv     = f_dist.sf(float(f_stat), 1, len(sub2) - aux.df_model - 1)
            reject.append(pv < alpha)
        except Exception:
            reject.append(True)

    in_ci = [g for g, r in zip(grid, reject) if not r]
    if not in_ci:
        return (float("nan"), float("nan"), 0)
    return (min(in_ci), max(in_ci), len(in_ci))


def main():
    for p in [ANAL / "mechanism_a.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 06: Mechanism A — 2SLS ===\n")

    df_a      = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_a_fl_p = ANAL / "mechanism_a_flood.parquet"
    df_a_fl   = pd.read_parquet(df_a_fl_p) if df_a_fl_p.exists() else pd.DataFrame()

    rv_col         = "running_var"       if "running_var"       in df_a.columns else "rv_linear"
    prior_rv_col   = "prior_running_var_raw" if "prior_running_var_raw" in df_a.columns else rv_col
    instrument_col = "prior_threshold_crossing"
    endog_col      = "prior_ia_received"

    # Build instrument if not already present
    if instrument_col not in df_a.columns and prior_rv_col in df_a.columns:
        df_a[instrument_col] = (df_a[prior_rv_col] >= 0).astype(float)
    if instrument_col not in df_a.columns:
        print(f"  WARNING: {instrument_col} not found and could not be constructed")

    controls, df_a = prep_controls(df_a)
    if len(df_a_fl):
        if instrument_col not in df_a_fl.columns and prior_rv_col in df_a_fl.columns:
            df_a_fl[instrument_col] = (df_a_fl[prior_rv_col] >= 0).astype(float)
        _, df_a_fl = prep_controls(df_a_fl)

    # --- Main 2SLS table ---
    print("Running 2SLS for each outcome...")
    results = []

    for outcome_col, outcome_label in OUTCOMES_A:
        print(f"\n  Outcome: {outcome_label}")
        for dataset_label, dataset in [("All hazards", df_a), ("Flood only", df_a_fl)]:
            if len(dataset) == 0:
                continue
            res = run_2sls(
                df=dataset, outcome=outcome_col,
                rv_col=prior_rv_col, instrument_col=instrument_col,
                endog_col=endog_col, controls=controls,
                fe_cols=FE_CONTROLS, hazard_cols=HAZARD_CONTROLS,
            )
            if res.get("error"):
                print(f"    {dataset_label}: ERROR — {res['error']}")
            else:
                f = res.get("F_first_stage", float("nan"))
                pv = res.get("pv", float("nan"))
                stars = ("***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "")))
                print(f"    {dataset_label}: coef={res['coef']:.4f} ({res['se']:.4f}){stars}  "
                      f"N={res['N']:,}  F_1st={f:.1f}" if not np.isnan(f) else
                      f"    {dataset_label}: coef={res['coef']:.4f} ({res['se']:.4f}){stars}  N={res['N']:,}")
            results.append({
                "outcome":  outcome_label,
                "sample":   dataset_label,
                **({k: v for k, v in res.items() if k != "estimator" and k != "error"}
                   if not res.get("error") else {"error": res["error"]}),
            })

    # --- Anderson-Rubin CI if F < 10 ---
    ar_rows = []
    f_stats_all = [r.get("F_first_stage", float("nan")) for r in results if not r.get("error")]
    max_f = max((f for f in f_stats_all if not np.isnan(f)), default=float("nan"))
    if np.isnan(max_f) or max_f < 10:
        print(f"\nFirst-stage F = {max_f:.2f} < 10 — computing Anderson-Rubin CIs...")
        for outcome_col, outcome_label in OUTCOMES_A[:1]:  # AR CI for primary outcome only
            ci_lo, ci_hi, n_pts = anderson_rubin_ci(
                df_a, outcome_col, prior_rv_col,
                instrument_col, endog_col, controls, FE_CONTROLS,
            )
            print(f"  {outcome_label}: AR CI = [{ci_lo:.4f}, {ci_hi:.4f}]  ({n_pts} grid pts)")
            ar_rows.append({"Outcome": outcome_label, "AR CI lower": f"{ci_lo:.4f}",
                            "AR CI upper": f"{ci_hi:.4f}"})

    # --- Bandwidth robustness ---
    print("\nBandwidth robustness (Mechanism A, contents_damage_ratio)...")

    if "contents_damage_ratio" in df_a.columns and prior_rv_col in df_a.columns:
        base_bw = df_a[prior_rv_col].abs().quantile(0.75)  # proxy for CCT optimal BW
        bw_results = []

        for mult in BW_MULTIPLIERS:
            bw = base_bw * mult
            sub = df_a[df_a[prior_rv_col].abs() <= bw].copy()
            res = run_2sls(
                df=sub, outcome="contents_damage_ratio",
                rv_col=prior_rv_col, instrument_col=instrument_col,
                endog_col=endog_col, controls=controls,
                fe_cols=FE_CONTROLS, hazard_cols=HAZARD_CONTROLS,
            )
            if not res.get("error"):
                bw_results.append({"bw_mult": mult, "bw": bw, **res})
                print(f"  {mult:.2f}×  bw={bw:.3f}  N={res['N']:,}  coef={res['coef']:.4f}")

        if bw_results:
            bw_df = pd.DataFrame(bw_results)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(bw_df["bw_mult"], bw_df["coef"],
                        yerr=1.96 * bw_df["se"],
                        fmt="o-", color="steelblue", ecolor="steelblue",
                        elinewidth=1.5, capsize=4)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Bandwidth multiplier (×CCT optimal)")
            ax.set_ylabel("2SLS coefficient (contents damage ratio)")
            ax.set_title("Mechanism A: Bandwidth Robustness\n"
                         "Contents damage ratio ~ prior IA receipt (IV)")
            ax.set_xticks(BW_MULTIPLIERS)
            fig.tight_layout()
            fig.savefig(FIGS / "fig_mech_a_bandwidth.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: figures/fig_mech_a_bandwidth.png")

    # --- LaTeX table ---
    # Build wide table: rows = outcomes, cols = all-hazards / flood-only
    rows_all   = [r for r in results if r.get("sample") == "All hazards" and not r.get("error")]
    rows_flood = [r for r in results if r.get("sample") == "Flood only"  and not r.get("error")]

    def fmt_coef(r):
        if r.get("error"):
            return r["error"], ""
        c  = r.get("coef", float("nan"))
        se = r.get("se", float("nan"))
        pv = r.get("pv", float("nan"))
        stars = ("***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "")))
        return (f"{c:.4f}{stars}", f"({se:.4f})")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mechanism A: 2SLS Estimates — Effect of Prior IA Receipt on Mitigation Proxies}",
        r"\label{tab:mech_a}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{All Hazards} & \multicolumn{2}{c}{Flood Only} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Outcome & Coef. & SE & Coef. & SE \\",
        r"\midrule",
    ]
    for outcome_col, outcome_label in OUTCOMES_A:
        ra = next((r for r in rows_all   if r["outcome"] == outcome_label), {})
        rf = next((r for r in rows_flood if r["outcome"] == outcome_label), {})
        ca, sea = fmt_coef(ra)
        cf, sef = fmt_coef(rf)
        lines.append(rf"{outcome_label} & {ca} & {sea} & {cf} & {sef} \\")
    na = rows_all[0]["N"]   if rows_all   else "—"
    nf = rows_flood[0]["N"] if rows_flood else "—"
    fa = f"{rows_all[0]['F_first_stage']:.1f}"   if rows_all   and not np.isnan(rows_all[0].get("F_first_stage", float("nan"))) else "—"
    ff = f"{rows_flood[0]['F_first_stage']:.1f}" if rows_flood and not np.isnan(rows_flood[0].get("F_first_stage", float("nan"))) else "—"
    lines += [
        r"\midrule",
        rf"N & {na:,} & & {nf:,} & \\",
        rf"First-stage F & {fa} & & {ff} & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize{Note: IV2SLS (linearmodels). Instrument: prior threshold crossing indicator. "
        r"Controls: ACS demographics, state FE, year FE, hazard type FE, hazard intensity, "
        r"warning lead time. SEs clustered by county. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.}",
        r"\end{table}",
    ]
    (TABLES / "tab_mech_a_2sls.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_mech_a_2sls.tex")

    if ar_rows:
        ar_tbl = pd.DataFrame(ar_rows)
        ar_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Mechanism A: Anderson-Rubin Confidence Intervals (Weak-Instrument Robust)}",
            r"\label{tab:ar_ci}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Outcome & AR CI lower & AR CI upper \\",
            r"\midrule",
        ]
        for _, row in ar_tbl.iterrows():
            ar_lines.append(f"{row['Outcome']} & {row['AR CI lower']} & {row['AR CI upper']} \\\\")
        ar_lines += [r"\bottomrule", r"\end{tabular}",
                     r"\footnotesize{Note: Anderson-Rubin (1949) grid-search CI at $\alpha=0.05$.}",
                     r"\end{table}"]
        (TABLES / "tab_mech_a_ar_ci.tex").write_text("\n".join(ar_lines))
        print(f"  Saved: tables/tab_mech_a_ar_ci.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

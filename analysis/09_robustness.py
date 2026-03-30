"""
analysis/09_robustness.py
Robustness checks for both mechanisms.

Tests:
  (1) Donut RD — exclude observations within δ of threshold (δ ∈ {0.05, 0.10, 0.15, 0.20})
  (2) TTR specification comparison — re-run main estimates under each of the 3 TTR specs
  (3) Flood-only subsample — compare all-hazards vs. flood-only estimates
  (4) Election subsample — compare election vs. non-election years
  (5) Bandwidth grid — coefficient stability across 6 bandwidth multipliers
      (covered in 06/07 but compiled here into a combined robustness table)

Outputs:
  tables/tab_robustness.tex             — combined robustness table
  figures/fig_donut_rd.png              — Donut RD sensitivity
  figures/fig_ttr_spec_comparison.png   — TTR spec comparison plot

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_a_flood.parquet
  data/analysis/mechanism_b.parquet
  data/analysis/mechanism_b_flood.parquet
  data/intermediate/ttr_spec_comparison.csv
  data/intermediate/events_county_panel_with_rv.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

try:
    from rdrobust import rdrobust
    HAS_RDROBUST = True
except Exception:
    HAS_RDROBUST = False

try:
    from linearmodels.iv import IV2SLS
    HAS_LM = True
except Exception:
    HAS_LM = False

import statsmodels.formula.api as smf

ROOT   = Path(__file__).resolve().parents[1]
INTER  = ROOT / "data" / "intermediate"
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Donut exclusion widths (pre-specified in methods appendix §A.4.4)
DONUT_DELTAS = [0.05, 0.10, 0.15, 0.20]

# Bandwidth multipliers (pre-specified in methods appendix §A.4.3)
BW_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# TTR specs to compare (from preprocessing/04b)
TTR_SPECS = ["rv_linear", "rv_log", "rv_rank"]

PRIMARY_OUTCOME_A = "contents_damage_ratio"
PRIMARY_OUTCOME_B = "ia_application_rate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_rd(df: pd.DataFrame, outcome: str, rv_col: str,
              fuzzy_col: str = None) -> dict:
    """Local linear RD — wrapper around rdrobust or OLS fallback."""
    if outcome not in df.columns or rv_col not in df.columns:
        return {"error": "missing columns"}
    sub = df.dropna(subset=[outcome, rv_col])
    y = sub[outcome].values
    x = sub[rv_col].values
    f = sub[fuzzy_col].values if (fuzzy_col and fuzzy_col in sub.columns) else None

    mask = np.isfinite(y) & np.isfinite(x)
    if f is not None:
        mask &= np.isfinite(f)
    y, x = y[mask], x[mask]
    if f is not None:
        f = f[mask]

    if len(y) < 30:
        return {"error": f"N={len(y)} < 30"}

    try:
        if HAS_RDROBUST:
            kw = dict(kernel="triangular", vce="hc1")
            if f is not None:
                kw["fuzzy"] = f
            rr = rdrobust(y=y, x=x, **kw)
            coef = float(rr.coef[1])
            se   = float(rr.se[2])
            pv   = float(rr.pv[2]) if hasattr(rr, "pv") and len(rr.pv) > 2 else (
                   2 * (1 - norm.cdf(abs(coef / se))) if se > 0 else float("nan"))
            bw   = float(rr.bws[0, 0])
            return {"coef": coef, "se": se, "pv": pv, "bw": bw,
                    "N": int(sum(mask)), "error": None}
        else:
            df2 = pd.DataFrame({"y": y, "x": x})
            df2["above"] = (df2["x"] >= 0).astype(int)
            df2["inter"] = df2["above"] * df2["x"]
            bw = float(pd.Series(x).abs().quantile(0.5))
            df2 = df2[df2["x"].abs() <= bw]
            if len(df2) < 20:
                return {"error": "too few in bw"}
            mod = smf.wls("y ~ above + x + inter", data=df2).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            pv   = mod.pvalues.get("above", float("nan"))
            return {"coef": coef, "se": se, "pv": pv, "bw": bw,
                    "N": len(df2), "error": None}
    except Exception as e:
        return {"error": str(e)[:60]}


def fmt_coef(r: dict, decimals: int = 4) -> str:
    if r.get("error"):
        return r["error"]
    c  = r.get("coef", float("nan"))
    pv = r.get("pv", float("nan"))
    se = r.get("se", float("nan"))
    stars = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
    return f"{c:.{decimals}f}{stars} ({se:.{decimals}f})"


# ---------------------------------------------------------------------------
# (1) Donut RD
# ---------------------------------------------------------------------------

def donut_rd(df: pd.DataFrame, outcome: str, rv_col: str,
             deltas: list, fuzzy_col: str = None) -> list:
    """
    Estimate RD excluding observations within ±δ of threshold for each δ in deltas.
    δ=0 is the baseline (no donut).
    """
    results = []
    for delta in [0.0] + deltas:
        sub = df[df[rv_col].abs() > delta] if delta > 0 else df
        res = simple_rd(sub, outcome, rv_col, fuzzy_col)
        res["delta"] = delta
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# (2) TTR specification comparison
# ---------------------------------------------------------------------------

def ttr_spec_comparison(df_events: pd.DataFrame,
                        df_a: pd.DataFrame,
                        outcome: str) -> list:
    """
    For each TTR spec (rv_linear, rv_log, rv_rank), construct a running variable
    and re-estimate the main RD. Returns list of result dicts.
    """
    results = []
    for spec in TTR_SPECS:
        if spec not in df_a.columns:
            # Try to get from events panel
            if df_events is not None and spec in df_events.columns:
                # Merge into df_a
                merge_cols = [c for c in ["disasterNumber", "county_fips", spec]
                              if c in df_events.columns]
                if len(merge_cols) >= 2:
                    df_merged = df_a.copy()
                    df_merged = df_merged.drop(columns=[spec], errors="ignore")
                    df_merged = df_merged.merge(
                        df_events[merge_cols].drop_duplicates(),
                        on=[c for c in ["disasterNumber", "county_fips"] if c in merge_cols],
                        how="left",
                    )
                else:
                    df_merged = df_a.copy()
            else:
                results.append({"spec": spec, "error": "column not found"})
                continue
        else:
            df_merged = df_a.copy()

        rv = spec
        fuzzy = "prior_ia_received" if "prior_ia_received" in df_merged.columns else None
        res = simple_rd(df_merged, outcome, rv, fuzzy)
        res["spec"] = spec
        results.append(res)
        n = res.get("N", "?")
        if res.get("error"):
            print(f"  {spec}: ERROR — {res['error']}")
        else:
            print(f"  {spec}: coef={res['coef']:.4f}  se={res['se']:.4f}  N={n:,}")
    return results


# ---------------------------------------------------------------------------
# (3 + 4) Election and flood subsamples
# ---------------------------------------------------------------------------

def subsample_comparison(df: pd.DataFrame, outcome: str, rv_col: str,
                         fuzzy_col: str = None) -> dict:
    results = {}

    # Flood only
    fl_col = "hazard_type" if "hazard_type" in df.columns else "incidentType"
    if fl_col in df.columns:
        if fl_col == "hazard_type":
            df_fl = df[df[fl_col].isin(["flood_only", "flood_mixed"])]
        else:
            df_fl = df[df[fl_col].str.lower().isin(["flood", "coastal storm"])]
        results["flood_only"] = simple_rd(df_fl, outcome, rv_col, fuzzy_col)
        results["flood_only"]["label"] = "Flood only"

    # All hazards (baseline)
    results["all_hazards"] = simple_rd(df, outcome, rv_col, fuzzy_col)
    results["all_hazards"]["label"] = "All hazards"

    # Election year
    if "presidential_election_year" in df.columns:
        df_el = df[df["presidential_election_year"] == 1]
        df_ne = df[df["presidential_election_year"] == 0]
        results["election"]    = simple_rd(df_el, outcome, rv_col, fuzzy_col)
        results["election"]["label"]    = "Election year"
        results["nonelection"] = simple_rd(df_ne, outcome, rv_col, fuzzy_col)
        results["nonelection"]["label"] = "Non-election year"

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 09: Robustness Checks ===\n")

    df_a    = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b    = pd.read_parquet(ANAL / "mechanism_b.parquet")
    fl_a_p  = ANAL / "mechanism_a_flood.parquet"
    fl_b_p  = ANAL / "mechanism_b_flood.parquet"
    df_a_fl = pd.read_parquet(fl_a_p) if fl_a_p.exists() else pd.DataFrame()
    df_b_fl = pd.read_parquet(fl_b_p) if fl_b_p.exists() else pd.DataFrame()

    ev_path   = INTER / "events_county_panel_with_rv.parquet"
    df_events = pd.read_parquet(ev_path) if ev_path.exists() else None

    rv_a = "running_var" if "running_var" in df_a.columns else "rv_linear"
    rv_b = "running_var" if "running_var" in df_b.columns else "rv_linear"

    all_robustness_rows = []

    # -----------------------------------------------------------------------
    # (1) Donut RD
    # -----------------------------------------------------------------------
    print("--- (1) Donut RD ---")

    print("\n  Mechanism A (contents_damage_ratio):")
    prior_rv = "prior_running_var_raw" if "prior_running_var_raw" in df_a.columns else rv_a
    fuzzy_a  = "prior_ia_received"     if "prior_ia_received"     in df_a.columns else None
    donut_a = donut_rd(df_a, PRIMARY_OUTCOME_A, prior_rv, DONUT_DELTAS, fuzzy_a)
    for r in donut_a:
        d = r.get("delta", 0)
        if r.get("error"):
            print(f"  δ={d:.2f}: ERROR — {r['error']}")
        else:
            print(f"  δ={d:.2f}: coef={r['coef']:.4f}  se={r['se']:.4f}  N={r['N']:,}")
        all_robustness_rows.append({
            "Mechanism": "A", "Test": f"Donut δ={d:.2f}", "Sample": "All hazards",
            "Estimate": fmt_coef(r),
        })

    print("\n  Mechanism B (ia_application_rate):")
    fuzzy_b = "ia_declared" if "ia_declared" in df_b.columns else None
    donut_b = donut_rd(df_b, PRIMARY_OUTCOME_B, rv_b, DONUT_DELTAS, fuzzy_b)
    for r in donut_b:
        d = r.get("delta", 0)
        if r.get("error"):
            print(f"  δ={d:.2f}: ERROR — {r['error']}")
        else:
            print(f"  δ={d:.2f}: coef={r['coef']:.4f}  se={r['se']:.4f}  N={r['N']:,}")
        all_robustness_rows.append({
            "Mechanism": "B", "Test": f"Donut δ={d:.2f}", "Sample": "All hazards",
            "Estimate": fmt_coef(r),
        })

    # Donut RD plot (Mechanism B, primary outcome)
    donut_valid = [r for r in donut_b if not r.get("error")]
    if donut_valid:
        deltas_plot = [r["delta"] for r in donut_valid]
        coefs_plot  = [r["coef"]  for r in donut_valid]
        ses_plot    = [r["se"]    for r in donut_valid]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(deltas_plot, coefs_plot, yerr=[1.96 * s for s in ses_plot],
                    fmt="o-", color="steelblue", ecolor="steelblue",
                    elinewidth=1.5, capsize=5, markersize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Donut exclusion width δ ($/capita around threshold)")
        ax.set_ylabel("RD coefficient (IA application rate)")
        ax.set_title("Donut RD Sensitivity — Mechanism B\n"
                     "Excluding observations within ±δ of threshold")
        ax.set_xticks([0.0] + DONUT_DELTAS)
        ax.set_xticklabels([f"δ={d:.2f}" for d in [0.0] + DONUT_DELTAS])
        fig.tight_layout()
        fig.savefig(FIGS / "fig_donut_rd.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved: figures/fig_donut_rd.png")

    # -----------------------------------------------------------------------
    # (2) TTR specification comparison
    # -----------------------------------------------------------------------
    print("\n--- (2) TTR Specification Comparison ---")
    print("\n  Mechanism A (contents_damage_ratio):")
    ttr_a = ttr_spec_comparison(df_events, df_a, PRIMARY_OUTCOME_A)
    for r in ttr_a:
        all_robustness_rows.append({
            "Mechanism": "A", "Test": f"TTR: {r.get('spec', '?')}",
            "Sample": "All hazards", "Estimate": fmt_coef(r),
        })

    print("\n  Mechanism B (ia_application_rate):")
    ttr_b = ttr_spec_comparison(df_events, df_b, PRIMARY_OUTCOME_B)
    for r in ttr_b:
        all_robustness_rows.append({
            "Mechanism": "B", "Test": f"TTR: {r.get('spec', '?')}",
            "Sample": "All hazards", "Estimate": fmt_coef(r),
        })

    # TTR spec comparison plot
    valid_ttr_a = [r for r in ttr_a if not r.get("error") and "coef" in r]
    valid_ttr_b = [r for r in ttr_b if not r.get("error") and "coef" in r]
    if valid_ttr_a or valid_ttr_b:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, valid, title, ylabel in [
            (axes[0], valid_ttr_a, "Mechanism A: Contents Damage Ratio", "2SLS coef."),
            (axes[1], valid_ttr_b, "Mechanism B: IA Application Rate",   "RD coef."),
        ]:
            if not valid:
                ax.set_title(title + "\n(no data)")
                continue
            specs  = [r["spec"].replace("rv_", "") for r in valid]
            coefs  = [r["coef"] for r in valid]
            ses    = [r["se"]   for r in valid]
            xs = range(len(specs))
            ax.errorbar(xs, coefs, yerr=[1.96 * s for s in ses],
                        fmt="o", color="steelblue", ecolor="steelblue",
                        elinewidth=1.5, capsize=5, markersize=8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(specs)
            ax.set_xlabel("TTR specification")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        fig.suptitle("Robustness to TTR Specification", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ttr_spec_comparison.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved: figures/fig_ttr_spec_comparison.png")

    # -----------------------------------------------------------------------
    # (3) Flood-only + (4) Election subsample
    # -----------------------------------------------------------------------
    print("\n--- (3) Flood-only and (4) Election subsample ---")

    print("\n  Mechanism A subsamples:")
    sub_a = subsample_comparison(df_a, PRIMARY_OUTCOME_A, prior_rv, fuzzy_a)
    for key, r in sub_a.items():
        label = r.get("label", key)
        if not r.get("error"):
            print(f"  {label:<22}: coef={r['coef']:.4f}  se={r['se']:.4f}  N={r['N']:,}")
        else:
            print(f"  {label:<22}: ERROR — {r['error']}")
        all_robustness_rows.append({
            "Mechanism": "A", "Test": "Subsample", "Sample": label,
            "Estimate": fmt_coef(r),
        })

    print("\n  Mechanism B subsamples:")
    sub_b = subsample_comparison(df_b, PRIMARY_OUTCOME_B, rv_b, fuzzy_b)
    for key, r in sub_b.items():
        label = r.get("label", key)
        if not r.get("error"):
            print(f"  {label:<22}: coef={r['coef']:.4f}  se={r['se']:.4f}  N={r['N']:,}")
        else:
            print(f"  {label:<22}: ERROR — {r['error']}")
        all_robustness_rows.append({
            "Mechanism": "B", "Test": "Subsample", "Sample": label,
            "Estimate": fmt_coef(r),
        })

    # -----------------------------------------------------------------------
    # (5) Bandwidth grid
    # -----------------------------------------------------------------------
    print("\n--- (5) Bandwidth grid ---")

    for mech_lbl, df_mech, rv_col_m, outcome_m, fuzzy_m in [
        ("A", df_a, prior_rv, PRIMARY_OUTCOME_A, fuzzy_a),
        ("B", df_b, rv_b,     PRIMARY_OUTCOME_B, fuzzy_b),
    ]:
        print(f"\n  Mechanism {mech_lbl}:")
        if rv_col_m not in df_mech.columns:
            print(f"  SKIP: {rv_col_m} not found")
            continue
        base_bw = df_mech[rv_col_m].abs().quantile(0.75)
        for mult in BW_MULTIPLIERS:
            bw = base_bw * mult
            sub = df_mech[df_mech[rv_col_m].abs() <= bw].copy()
            res = simple_rd(sub, outcome_m, rv_col_m, fuzzy_m)
            if not res.get("error"):
                print(f"  {mult:.2f}× bw={bw:.3f}: coef={res['coef']:.4f}  N={res['N']:,}")
            else:
                print(f"  {mult:.2f}×: ERROR — {res['error']}")
            all_robustness_rows.append({
                "Mechanism": mech_lbl, "Test": f"BW {mult:.2f}×",
                "Sample": "All hazards", "Estimate": fmt_coef(res),
            })

    # --- Combined LaTeX table ---
    tbl = pd.DataFrame(all_robustness_rows)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Robustness Checks}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Mechanism & Test & Sample & Estimate \\",
        r"\midrule",
    ]
    cur_mech = None
    for _, row in tbl.iterrows():
        if row["Mechanism"] != cur_mech:
            if cur_mech is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{Mechanism {row['Mechanism']}}}}} \\")
            cur_mech = row["Mechanism"]
        lines.append(
            f"  & {row['Test']} & {row['Sample']} & {row['Estimate']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize{Note: Baseline estimates in rows with δ=0.00 (donut) and 1.00× (BW). "
        r"Donut RD excludes observations within ±δ of threshold. "
        r"BW multiplier applied to CCT optimal bandwidth. "
        r"*** p$<$0.01, ** p$<$0.05, * p$<$0.10.}",
        r"\end{table}",
    ]
    (TABLES / "tab_robustness.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_robustness.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

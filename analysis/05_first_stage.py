"""
analysis/05_first_stage.py
First-stage diagnostics: binscatter plots and F-statistics by subgroup.

Produces:
  (1) Binscatter of IA declared vs. running variable (discontinuity visualization)
  (2) First-stage F by TTR specification, years-since-prior bin, election year, hazard type
  (3) Election-year interaction model (methods appendix §F.2)

Outputs:
  figures/fig_binscatter_a.png         — Mechanism A binscatter
  figures/fig_binscatter_b.png         — Mechanism B binscatter
  figures/fig_first_stage_subgroups.png — F-stat by subgroup
  tables/tab_first_stage.tex           — First-stage table

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_b.parquet
  data/intermediate/ttr_spec_comparison.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from rdrobust import rdrobust
    HAS_RDROBUST = True
except Exception:
    HAS_RDROBUST = False

import statsmodels.formula.api as smf

ROOT   = Path(__file__).resolve().parents[1]
INTER  = ROOT / "data" / "intermediate"
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Temporal decay bins (pre-specified in project plan §temporal decay / methods appendix §G.2)
TEMPORAL_BINS   = [0, 2, 5, 9, 15, float("inf")]
TEMPORAL_LABELS = ["0–2yr", "3–5yr", "6–9yr", "10–15yr", "15+yr"]


# ---------------------------------------------------------------------------
# Binscatter
# ---------------------------------------------------------------------------

def binscatter(df: pd.DataFrame, rv_col: str, outcome_col: str,
               n_bins: int = 20, title: str = "", outpath: Path = None) -> None:
    if rv_col not in df.columns or outcome_col not in df.columns:
        return

    sub = df[[rv_col, outcome_col]].dropna()
    x   = sub[rv_col].values
    y   = sub[outcome_col].values

    # Trim to ±2 IQR
    q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q75 - q25
    lo, hi = max(x.min(), q25 - 2 * iqr), min(x.max(), q75 + 2 * iqr)
    mask = (x >= lo) & (x <= hi)
    x, y = x[mask], y[mask]

    # Equal-count bins on each side
    xl, yl = x[x < 0], y[x < 0]
    xr, yr = x[x >= 0], y[x >= 0]

    def bin_means(xv, yv, nb):
        if len(xv) < nb:
            return np.array([]), np.array([])
        edges = np.percentile(xv, np.linspace(0, 100, nb + 1))
        mids, ymeans = [], []
        for i in range(nb):
            m = (xv >= edges[i]) & (xv < edges[i + 1])
            if m.sum() > 0:
                mids.append(xv[m].mean())
                ymeans.append(yv[m].mean())
        return np.array(mids), np.array(ymeans)

    nb = min(n_bins, max(5, len(xl) // 5))
    ml, yl_m = bin_means(xl, yl, nb)
    mr, yr_m = bin_means(xr, yr, nb)

    # Local linear fit on binned data
    def ll_fit(mx, my):
        if len(mx) < 3:
            return mx, my
        from numpy.polynomial.polynomial import polyfit, polyval
        c = polyfit(mx, my, 1)
        return mx, polyval(mx, c)

    xl_f, yl_f = ll_fit(ml, yl_m)
    xr_f, yr_f = ll_fit(mr, yr_m)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ml, yl_m, color="steelblue", s=25, zorder=3, label="Bin means (left)")
    ax.scatter(mr, yr_m, color="steelblue", s=25, zorder=3, label="Bin means (right)")
    ax.plot(xl_f, yl_f, color="navy",    linewidth=2.0)
    ax.plot(xr_f, yr_f, color="navy",    linewidth=2.0)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Threshold")

    ax.set_xlabel("Running variable (TTR-adjusted per-capita damage − threshold, $/capita)")
    ax.set_ylabel(outcome_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"  Saved: {outpath.relative_to(ROOT)}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# First-stage F computation (OLS-based diagnostic)
# ---------------------------------------------------------------------------

def first_stage_f(df: pd.DataFrame, rv_col: str,
                  outcome_col: str = "ia_declared",
                  state_col: str = "state_fips",
                  year_col: str = "event_year") -> float:
    """
    OLS first stage: outcome ~ 1[rv >= 0] + rv + state FE + year FE.
    Returns F-stat = (coef_above / se_above)^2.
    """
    cols = [rv_col, outcome_col]
    for c in [state_col, year_col]:
        if c in df.columns:
            cols.append(c)
    sub = df[cols].dropna()
    if len(sub) < 30:
        return float("nan")
    sub2 = sub.copy()
    sub2["above"] = (sub2[rv_col] >= 0).astype(int)
    fe_terms = ""
    if state_col in sub2.columns:
        fe_terms += f" + C({state_col})"
    if year_col in sub2.columns:
        fe_terms += f" + C({year_col})"
    formula = f"{outcome_col} ~ above + {rv_col}{fe_terms}"
    try:
        mod  = smf.ols(formula, data=sub2).fit()
        coef = mod.params.get("above", float("nan"))
        se   = mod.bse.get("above", float("nan"))
        return float((coef / se) ** 2) if se > 0 else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 05: First Stage ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    rv_a = "running_var" if "running_var" in df_a.columns else "rv_linear"
    rv_b = "running_var" if "running_var" in df_b.columns else "rv_linear"

    # --- Binscatter plots ---
    print("Generating binscatter plots...")

    # Mechanism A: prior_ia_received vs. prior running variable
    prior_rv = "prior_running_var_raw" if "prior_running_var_raw" in df_a.columns else rv_a
    binscatter(df_a, prior_rv, "prior_ia_received",
               title="First Stage — Mechanism A: Prior IA receipt vs. prior running variable",
               outpath=FIGS / "fig_binscatter_a.png")

    # Mechanism B: ia_declared vs. contemporaneous running variable
    binscatter(df_b, rv_b, "ia_declared",
               title="First Stage — Mechanism B: IA declaration vs. contemporaneous running variable",
               outpath=FIGS / "fig_binscatter_b.png")

    # --- F-statistics by TTR specification ---
    print("\nFirst-stage F by TTR specification (from ttr_spec_comparison.csv):")
    comp_path = INTER / "ttr_spec_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        print(comp[["spec", "f_stat", "n", "coef", "se"]].to_string(index=False))
    else:
        print("  ttr_spec_comparison.csv not found")
    comp_data = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

    # --- F-statistics by years-since-prior bin ---
    print("\nFirst-stage F by years-since-prior bin (Mechanism A):")
    f_by_bin = []
    if "years_since_prior" in df_a.columns:
        df_a2 = df_a.copy()
        df_a2["bin"] = pd.cut(df_a2["years_since_prior"],
                              bins=TEMPORAL_BINS, labels=TEMPORAL_LABELS, right=True)
        for lbl in TEMPORAL_LABELS:
            subset = df_a2[df_a2["bin"] == lbl]
            f = first_stage_f(subset, rv_a)
            n = len(subset)
            flag = " ← F < 10" if not np.isnan(f) and f < 10 else ""
            print(f"  {lbl:<10}  N={n:,}  F={f:.2f}{flag}" if not np.isnan(f) else
                  f"  {lbl:<10}  N={n:,}  F=N/A")
            f_by_bin.append({"bin": lbl, "N": n, "F": f})
    f_by_bin_df = pd.DataFrame(f_by_bin)

    # --- F-statistics by election year ---
    print("\nFirst-stage F by election year (Mechanism A):")
    f_election_rows = []
    if "presidential_election_year" in df_a.columns:
        for elec_val, label in [(1, "Election year"), (0, "Non-election year")]:
            subset = df_a[df_a["presidential_election_year"] == elec_val]
            f = first_stage_f(subset, rv_a)
            n = len(subset)
            print(f"  {label:<20}  N={n:,}  F={f:.2f}" if not np.isnan(f) else
                  f"  {label:<20}  N={n:,}  F=N/A")
            f_election_rows.append({"subgroup": label, "N": n, "F": f})

    # --- Election interaction model ---
    print("\nElection-year interaction model:")
    if all(c in df_a.columns for c in [rv_a, "ia_declared", "presidential_election_year"]):
        sub = df_a[[rv_a, "ia_declared", "presidential_election_year",
                    "state_fips", "event_year"]].dropna().copy()
        sub["above"]       = (sub[rv_a] >= 0).astype(int)
        sub["above_elec"]  = sub["above"] * sub["presidential_election_year"]
        fe = " + C(state_fips) + C(event_year)" if "state_fips" in sub.columns else ""
        try:
            mod = smf.ols(
                f"ia_declared ~ above + presidential_election_year + above_elec + {rv_a}{fe}",
                data=sub
            ).fit()
            for param in ["above", "presidential_election_year", "above_elec"]:
                if param in mod.params:
                    c  = mod.params[param]
                    se = mod.bse[param]
                    pv = mod.pvalues[param]
                    stars = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
                    print(f"  {param:<32}: coef={c:.4f}  se={se:.4f}  p={pv:.4f} {stars}")
        except Exception as e:
            print(f"  Election interaction model failed: {e}")

    # --- F-statistics by hazard type ---
    print("\nFirst-stage F by hazard type (Mechanism A):")
    hazard_col = "hazard_type" if "hazard_type" in df_a.columns else "incidentType"
    f_hazard_rows = []
    if hazard_col in df_a.columns:
        for htype in df_a[hazard_col].dropna().unique():
            subset = df_a[df_a[hazard_col] == htype]
            if len(subset) < 30:
                continue
            f = first_stage_f(subset, rv_a)
            n = len(subset)
            print(f"  {str(htype):<20}  N={n:,}  F={f:.2f}" if not np.isnan(f) else
                  f"  {str(htype):<20}  N={n:,}  F=N/A")
            f_hazard_rows.append({"hazard_type": htype, "N": n, "F": f})

    # --- Subgroup F plot ---
    print("\nGenerating subgroup F-stat plot...")
    all_f_rows = []
    for r in f_by_bin:
        all_f_rows.append({"label": f"Bin: {r['bin']}", "F": r["F"], "N": r["N"],
                            "group": "Years since prior"})
    for r in f_election_rows:
        all_f_rows.append({"label": r["subgroup"], "F": r["F"], "N": r["N"],
                            "group": "Election year"})
    for r in f_hazard_rows:
        all_f_rows.append({"label": f"Hazard: {r['hazard_type']}", "F": r["F"], "N": r["N"],
                            "group": "Hazard type"})

    if any(not np.isnan(r["F"]) for r in all_f_rows):
        plot_rows = [r for r in all_f_rows if not np.isnan(r["F"])]
        labels = [r["label"] for r in plot_rows]
        f_vals = [r["F"] for r in plot_rows]

        fig, ax = plt.subplots(figsize=(8, max(4, len(plot_rows) * 0.5)))
        colors = []
        for r in plot_rows:
            colors.append("steelblue" if r["F"] >= 10 else "firebrick")
        ys = range(len(plot_rows))
        bars = ax.barh(list(ys), f_vals, color=colors, alpha=0.8, height=0.6)
        ax.axvline(10, color="black", linewidth=1.5, linestyle="--",
                   label="F = 10 (Staiger-Stock threshold)")
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("First-stage F-statistic")
        ax.set_title("First-Stage Instrument Strength by Subgroup\n(blue = F≥10, red = F<10)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_first_stage_subgroups.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: figures/fig_first_stage_subgroups.png")

    # --- LaTeX summary table ---
    table_sections = []
    if not comp_data.empty and "f_stat" in comp_data.columns:
        table_sections.append((
            "TTR Specification",
            comp_data.rename(columns={"spec": "Subgroup", "n": "N", "f_stat": "F-stat"}
            )[["Subgroup", "N", "F-stat"]].to_dict("records")
        ))
    if f_by_bin_df is not None and len(f_by_bin_df):
        table_sections.append((
            "Years Since Prior Event",
            [{"Subgroup": r["bin"], "N": r["N"], "F-stat": f"{r['F']:.2f}" if not np.isnan(r["F"]) else ""}
             for _, r in f_by_bin_df.iterrows()]
        ))
    if f_election_rows:
        table_sections.append((
            "Election Year",
            [{"Subgroup": r["subgroup"], "N": r["N"], "F-stat": f"{r['F']:.2f}" if not np.isnan(r["F"]) else ""}
             for r in f_election_rows]
        ))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{First-Stage F-Statistics by Subgroup}",
        r"\label{tab:first_stage}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Subgroup & N & F-statistic \\",
        r"\midrule",
    ]
    for section_name, section_rows in table_sections:
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{section_name}}}}} \\")
        for row in section_rows:
            n_str = f"{int(row['N']):,}" if str(row['N']).isdigit() else str(row['N'])
            f_str = str(row["F-stat"])
            lines.append(rf"  {row['Subgroup']} & {n_str} & {f_str} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize{Note: F = (coef/SE)$^2$ from OLS first stage with state and year FE. "
        r"F $\geq$ 10 indicates adequate instrument strength (Staiger and Stock 1997).}",
        r"\end{table}",
    ]
    (TABLES / "tab_first_stage.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_first_stage.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
analysis/01_descriptive_stats.py
Summary statistics and running-variable histograms.

Outputs (tables/ and figures/):
  tables/tab_desc_stats_a.tex       — Mechanism A summary by IA declared
  tables/tab_desc_stats_b.tex       — Mechanism B summary by IA declared
  figures/fig_rv_histogram_a.png    — Mechanism A running variable histogram
  figures/fig_rv_histogram_b.png    — Mechanism B running variable histogram

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_b.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Variables to include in summary table
SUMMARY_VARS_A = [
    ("contents_damage_ratio",         "Contents damage ratio"),
    ("building_paid_per_depth_foot",  "Building paid per flood foot"),
    ("total_claim_severity",          "Total claim severity ($)"),
    ("warning_lead_hours",            "Warning lead time (hours)"),
    ("years_since_prior",             "Years since prior event"),
    ("county_population",             "County population"),
    ("median_household_income",       "Median household income ($)"),
    ("pct_owner_occ",                 "Pct. owner-occupied (%)"),
    ("county_presidential_vote_margin", "Presidential vote margin"),
]

SUMMARY_VARS_B = [
    ("ia_application_rate",           "IA application rate"),
    ("ia_approval_rate",              "IA approval rate"),
    ("avg_ia_award_owner",            "Avg. IA award — owners ($)"),
    ("avg_ia_award_renter",           "Avg. IA award — renters ($)"),
    ("n_nfip_claims",                 "NFIP claims (ZIP)"),
    ("n_total_registrations",         "IA registrations (ZIP)"),
    ("county_population",             "County population"),
    ("median_household_income",       "Median household income ($)"),
    ("pct_owner_occ",                 "Pct. owner-occupied (%)"),
]


def fmt(val, decimals: int = 3) -> str:
    if pd.isna(val):
        return ""
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:.{decimals}f}"


def build_summary_table(df: pd.DataFrame, var_list: list) -> pd.DataFrame:
    """
    Build a summary table with columns:
      Variable, N (declared), Mean (declared), SD (declared),
               N (not decl.), Mean (not decl.), SD (not decl.), p-value
    """
    if "ia_declared" not in df.columns:
        df = df.copy()
        df["ia_declared"] = float("nan")

    decl     = df[df["ia_declared"] == 1]
    not_decl = df[df["ia_declared"] == 0]

    rows = []
    for col, label in var_list:
        if col not in df.columns:
            continue
        d_vals  = decl[col].dropna()
        nd_vals = not_decl[col].dropna()

        # Two-sample t-test for means
        from scipy import stats as scipy_stats
        pv = float("nan")
        if len(d_vals) >= 2 and len(nd_vals) >= 2:
            try:
                _, pv = scipy_stats.ttest_ind(d_vals, nd_vals, equal_var=False)
            except Exception:
                pass

        rows.append({
            "Variable":         label,
            "N_decl":           len(d_vals),
            "Mean_decl":        fmt(d_vals.mean()),
            "SD_decl":          fmt(d_vals.std()),
            "N_notdecl":        len(nd_vals),
            "Mean_notdecl":     fmt(nd_vals.mean()),
            "SD_notdecl":       fmt(nd_vals.std()),
            "p_value":          f"{pv:.3f}" if not (np.isnan(pv) if isinstance(pv, float) else False) else "",
        })

    return pd.DataFrame(rows)


def to_latex(tbl: pd.DataFrame, caption: str, label: str) -> str:
    ncol = len(tbl.columns)
    col_fmt = "l" + "r" * (ncol - 1)
    header = " & ".join(tbl.columns) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def plot_rv_histogram(df: pd.DataFrame, rv_col: str, title: str, outpath: Path) -> None:
    if rv_col not in df.columns:
        print(f"  WARNING: {rv_col} not in dataframe — skipping histogram")
        return

    vals = df[rv_col].dropna()
    # Trim to ±3 IQR for display
    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q75 - q25
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    vals_trim = vals[(vals >= lo) & (vals <= hi)]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram with 40 bins
    n_bins = 40
    ax.hist(vals_trim, bins=n_bins, color="steelblue", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Threshold (RV=0)")

    # IA declared / not declared vertical means
    if "ia_declared" in df.columns:
        mean_decl = vals[df.loc[vals.index, "ia_declared"] == 1].mean()
        mean_nd   = vals[df.loc[vals.index, "ia_declared"] == 0].mean()
        if not np.isnan(mean_decl):
            ax.axvline(mean_decl,    color="firebrick",  linewidth=1.2,
                       linestyle=":", label=f"Mean (IA declared): {mean_decl:.2f}")
        if not np.isnan(mean_nd):
            ax.axvline(mean_nd, color="forestgreen", linewidth=1.2,
                       linestyle=":", label=f"Mean (not declared): {mean_nd:.2f}")

    ax.set_xlabel("Running variable (TTR-adjusted per-capita damage − threshold, $/capita)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath.relative_to(ROOT)}")


def main():
    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 01: Descriptive Statistics ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    # --- Summary tables ---
    print("Building summary table: Mechanism A...")
    tbl_a = build_summary_table(df_a, SUMMARY_VARS_A)
    tbl_a.columns = [
        "Variable",
        "N (IA=1)", "Mean (IA=1)", "SD (IA=1)",
        "N (IA=0)", "Mean (IA=0)", "SD (IA=0)",
        "p-value",
    ]
    tbl_a.to_csv(TABLES / "tab_desc_stats_a.csv", index=False)
    latex_a = to_latex(
        tbl_a,
        caption="Descriptive Statistics — Mechanism A (Historical IV)",
        label="tab:desc_a",
    )
    (TABLES / "tab_desc_stats_a.tex").write_text(latex_a)
    print(f"  Saved: tables/tab_desc_stats_a.tex ({len(tbl_a)} rows)")

    print("Building summary table: Mechanism B...")
    tbl_b = build_summary_table(df_b, SUMMARY_VARS_B)
    tbl_b.columns = [
        "Variable",
        "N (IA=1)", "Mean (IA=1)", "SD (IA=1)",
        "N (IA=0)", "Mean (IA=0)", "SD (IA=0)",
        "p-value",
    ]
    tbl_b.to_csv(TABLES / "tab_desc_stats_b.csv", index=False)
    latex_b = to_latex(
        tbl_b,
        caption="Descriptive Statistics — Mechanism B (Contemporaneous RD)",
        label="tab:desc_b",
    )
    (TABLES / "tab_desc_stats_b.tex").write_text(latex_b)
    print(f"  Saved: tables/tab_desc_stats_b.tex ({len(tbl_b)} rows)")

    # --- Running variable histograms ---
    print("\nPlotting running variable histograms...")

    rv_a = "running_var" if "running_var" in df_a.columns else "rv_linear"
    rv_b = "running_var" if "running_var" in df_b.columns else "rv_linear"

    plot_rv_histogram(
        df_a, rv_a,
        "Running Variable Distribution — Mechanism A (Historical IV)",
        FIGS / "fig_rv_histogram_a.png",
    )
    plot_rv_histogram(
        df_b, rv_b,
        "Running Variable Distribution — Mechanism B (Contemporaneous RD)",
        FIGS / "fig_rv_histogram_b.png",
    )

    # --- Quick diagnostics ---
    print("\n--- Quick Dataset Diagnostics ---")
    for name, df, rv in [("Mechanism A", df_a, rv_a), ("Mechanism B", df_b, rv_b)]:
        n = len(df)
        ia_share = df["ia_declared"].mean() if "ia_declared" in df.columns else float("nan")
        rv_vals  = df[rv].dropna() if rv in df.columns else pd.Series(dtype=float)
        print(f"\n  {name}")
        print(f"    N={n:,}  IA declared={ia_share:.1%}")
        if len(rv_vals):
            print(f"    RV: mean={rv_vals.mean():.3f}  sd={rv_vals.std():.3f}  "
                  f"p25={rv_vals.quantile(.25):.3f}  p75={rv_vals.quantile(.75):.3f}")
            pct_above = (rv_vals >= 0).mean()
            print(f"    RV ≥ 0: {pct_above:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()

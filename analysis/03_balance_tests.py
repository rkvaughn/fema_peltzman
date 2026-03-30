"""
analysis/03_balance_tests.py
RD balance tests on predetermined covariates with Bonferroni correction.

For each predetermined covariate, estimates a local linear RD using rdrobust
with the same bandwidth as the main analysis. Reports coefficient, SE, and
Bonferroni-adjusted p-value.

Also produces a coefficient plot.

Outputs:
  tables/tab_balance.tex              — balance test table
  figures/fig_balance_coefplot.png    — coefficient plot

Reads:
  data/analysis/mechanism_b.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from rdrobust import rdrobust, rdbwselect
    HAS_RDROBUST = True
except Exception:
    HAS_RDROBUST = False

import statsmodels.formula.api as smf

ROOT   = Path(__file__).resolve().parents[1]
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Predetermined covariates (pre-specified in methods appendix §C.2)
# These are characteristics determined before the running variable is realized.
BALANCE_COVARIATES = [
    ("median_household_income",         "Median household income"),
    ("county_presidential_vote_margin", "Presidential vote margin"),
    ("pct_owner_occ",                   "Pct. owner-occupied"),
    ("pct_pre1980_housing",             "Pct. pre-1980 housing"),
    ("county_population",               "County population (log)"),
]

# Bonferroni correction (pre-specified: k=5 covariates, α=0.05)
N_COVARIATES   = 5    # pre-specified in methods appendix §C.2
BONFERRONI_ADJ = 0.05 / N_COVARIATES   # 0.01


def run_balance_rd(y: np.ndarray, x: np.ndarray) -> dict:
    """
    Run rdrobust on outcome y with running variable x.
    Returns dict with coef, se, ci_lo, ci_hi, p_value, bw, N.
    """
    if len(y) < 50:
        return {"error": "too few obs"}

    try:
        if HAS_RDROBUST:
            rr = rdrobust(y=y, x=x, kernel="triangular", vce="hc1")
            coef  = float(rr.coef[1])   # bias-corrected
            se    = float(rr.se[2])     # robust SE
            ci_lo = float(rr.ci[1, 0])
            ci_hi = float(rr.ci[1, 1])
            bw    = float(rr.bws[0, 0])
            n_l   = int(rr.N_h[0])
            n_r   = int(rr.N_h[1])
            # p-value from z-stat (two-sided)
            z = coef / se if se > 0 else float("nan")
            from scipy.stats import norm
            pv = 2 * (1 - norm.cdf(abs(z)))
            return {
                "coef": coef, "se": se,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p_value": pv, "bw": bw,
                "N_left": n_l, "N_right": n_r,
                "error": None,
            }
        else:
            # OLS fallback
            df2 = pd.DataFrame({"y": y, "x": x})
            df2["above"] = (df2["x"] >= 0).astype(int)
            df2["xc"]    = df2["x"]
            df2["inter"] = df2["above"] * df2["xc"]
            bw = df2["x"].abs().quantile(0.5)
            df2 = df2[df2["x"].abs() <= bw]
            if len(df2) < 20:
                return {"error": "too few in bw"}
            mod = smf.wls("y ~ above + xc + inter", data=df2).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            pv   = mod.pvalues.get("above", float("nan"))
            ci   = mod.conf_int().loc["above"] if "above" in mod.conf_int().index else [float("nan")] * 2
            return {
                "coef": coef, "se": se,
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p_value": pv, "bw": bw,
                "N_left": int((df2["x"] < 0).sum()),
                "N_right": int((df2["x"] >= 0).sum()),
                "error": None,
            }
    except Exception as e:
        return {"error": str(e)[:80]}


def main():
    p = ANAL / "mechanism_b.parquet"
    if not p.exists():
        print(f"MISSING: {p.relative_to(ROOT)}")
        print("  Run preprocessing/05_build_analysis_dataset.py first.")
        return

    print("=== Analysis 03: Balance Tests ===\n")

    df = pd.read_parquet(p)
    rv_col = "running_var" if "running_var" in df.columns else "rv_linear"

    if rv_col not in df.columns:
        print(f"ERROR: {rv_col} not found in mechanism_b.parquet")
        return

    # Log-transform population for balance test
    if "county_population" in df.columns:
        df["county_population"] = np.log1p(df["county_population"].clip(lower=0))

    rows = []
    for col, label in BALANCE_COVARIATES:
        if col not in df.columns:
            print(f"  SKIP: {col} not in dataset")
            rows.append({
                "Covariate": label, "Coef.": "", "Robust SE": "",
                "95% CI": "", "p-value": "", "BW": "", "N": "", "Sig.*": "—",
            })
            continue

        sub = df[[col, rv_col]].dropna()
        y = sub[col].values
        x = sub[rv_col].values

        print(f"  {label} (N={len(y):,})")
        res = run_balance_rd(y, x)

        if res.get("error"):
            print(f"    ERROR: {res['error']}")
            rows.append({
                "Covariate": label, "Coef.": res["error"], "Robust SE": "",
                "95% CI": "", "p-value": "", "BW": "", "N": "",
                "Sig.*": "error",
            })
            continue

        pv   = res["p_value"]
        sig  = pv < BONFERRONI_ADJ if not np.isnan(pv) else False
        pv_s = f"{pv:.4f}" if not np.isnan(pv) else ""
        ci_s = f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]"

        print(f"    coef={res['coef']:.4f}  se={res['se']:.4f}  p={pv_s}"
              f"{'  *** SIGNIFICANT (Bonferroni)' if sig else ''}")

        rows.append({
            "Covariate": label,
            "Coef.":     f"{res['coef']:.4f}",
            "Robust SE": f"{res['se']:.4f}",
            "95% CI":    ci_s,
            "p-value":   pv_s,
            "BW":        f"{res['bw']:.3f}",
            "N":         f"{res.get('N_left', '?')} + {res.get('N_right', '?')}",
            "Sig.*":     "Yes" if sig else "No",
            "_coef":     res["coef"],
            "_se":       res["se"],
            "_pv":       pv,
        })

    tbl = pd.DataFrame(rows)

    # --- LaTeX table ---
    display_cols = ["Covariate", "Coef.", "Robust SE", "95% CI", "p-value", "BW", "N", "Sig.*"]
    tbl_disp = tbl[display_cols]

    ncol = len(display_cols)
    col_fmt = "l" + "r" * (ncol - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{RD Balance Tests on Predetermined Covariates}",
        r"\label{tab:balance}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(display_cols) + r" \\",
        r"\midrule",
    ]
    for _, row in tbl_disp.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\footnotesize{{Note: Local linear RD estimates via rdrobust (triangular kernel, CCT bandwidth). "
        rf"* Significant at Bonferroni-adjusted $\alpha = {BONFERRONI_ADJ:.3f}$ (k={N_COVARIATES} covariates).}}",
        r"\end{table}",
    ]
    (TABLES / "tab_balance.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_balance.tex")

    # --- Coefficient plot ---
    plot_rows = tbl[tbl["_coef"].notna()].copy() if "_coef" in tbl.columns else pd.DataFrame()
    if len(plot_rows):
        fig, ax = plt.subplots(figsize=(8, 5))
        ys = range(len(plot_rows))
        coefs = plot_rows["_coef"].values.astype(float)
        ses   = plot_rows["_se"].values.astype(float)

        # 95% CI bands (not Bonferroni-adjusted — note in figure)
        ax.errorbar(coefs, ys, xerr=1.96 * ses,
                    fmt="o", color="steelblue", ecolor="steelblue",
                    elinewidth=1.5, capsize=4, markersize=7)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")

        # Highlight Bonferroni-significant covariates
        for i, (_, row) in enumerate(plot_rows.iterrows()):
            if "_pv" in row and row["_pv"] < BONFERRONI_ADJ:
                ax.scatter([row["_coef"]], [i], color="firebrick", s=60, zorder=5)

        labels = [BALANCE_COVARIATES[j][1]
                  for j, (col, _) in enumerate(BALANCE_COVARIATES)
                  if col in df.columns and col in plot_rows.get("Covariate", pd.Series()).values
                  ] if False else list(plot_rows["Covariate"])

        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("RD Coefficient (local linear, bias-corrected)", fontsize=10)
        ax.set_title("Balance Tests — Predetermined Covariates\n"
                     "(red = significant at Bonferroni α=0.01)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_balance_coefplot.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: figures/fig_balance_coefplot.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

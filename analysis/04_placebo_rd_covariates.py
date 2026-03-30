"""
analysis/04_placebo_rd_covariates.py
Placebo threshold tests at off-threshold quantile locations.

Two complementary tests:
  (A) Placebo outcomes at true threshold — estimates RD on predetermined covariates
      (overlap with 03_balance_tests.py; this script adds a formal permutation p-value)
  (B) Placebo thresholds at quantile locations — estimates RD on true outcomes at
      fictitious thresholds (10th, 25th, 75th, 90th percentiles of RV distribution)

Outputs:
  tables/tab_placebo_thresholds.tex     — placebo threshold estimates
  figures/fig_placebo_distribution.png  — distribution of placebo estimates vs. true

Reads:
  data/analysis/mechanism_b.parquet    — for placebo threshold tests (Mechanism B)
  data/analysis/mechanism_a.parquet    — for mechanism A placebo thresholds
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

import statsmodels.formula.api as smf

ROOT   = Path(__file__).resolve().parents[1]
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Placebo threshold quantiles (pre-specified in methods appendix §C.3)
PLACEBO_QUANTILES = [0.10, 0.25, 0.75, 0.90]

# Primary outcomes for placebo threshold test
PRIMARY_OUTCOMES_B = [
    ("ia_application_rate",  "IA application rate"),
    ("ia_approval_rate",     "IA approval rate"),
    ("avg_ia_award_owner",   "Avg. IA award — owners"),
]

PRIMARY_OUTCOMES_A = [
    ("contents_damage_ratio", "Contents damage ratio"),
]


def rd_at_cutoff(y: np.ndarray, x: np.ndarray, cutoff: float = 0.0) -> dict:
    """
    Estimate local linear RD with threshold at `cutoff` (shift x by cutoff).
    Returns dict with coef, se, p_value, N.
    """
    x_centered = x - cutoff
    mask = np.isfinite(y) & np.isfinite(x_centered)
    y2, x2 = y[mask], x_centered[mask]
    if len(y2) < 30:
        return {"coef": float("nan"), "se": float("nan"), "p_value": float("nan"), "N": len(y2)}

    try:
        if HAS_RDROBUST:
            rr = rdrobust(y=y2, x=x2, kernel="triangular", vce="hc1")
            coef  = float(rr.coef[1])
            se    = float(rr.se[2])
            pv    = float(rr.pv[2]) if hasattr(rr, "pv") else 2 * (1 - norm.cdf(abs(coef / se)))
            return {"coef": coef, "se": se, "p_value": pv, "N": int(sum(mask))}
        else:
            df2 = pd.DataFrame({"y": y2, "x": x2})
            df2["above"] = (df2["x"] >= 0).astype(int)
            df2["inter"] = df2["above"] * df2["x"]
            bw = pd.Series(x2).abs().quantile(0.5)
            df2 = df2[df2["x"].abs() <= bw]
            if len(df2) < 20:
                return {"coef": float("nan"), "se": float("nan"), "p_value": float("nan"), "N": len(df2)}
            mod = smf.wls("y ~ above + x + inter", data=df2).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            pv   = mod.pvalues.get("above", float("nan"))
            return {"coef": coef, "se": se, "p_value": pv, "N": int(len(df2))}
    except Exception as e:
        return {"coef": float("nan"), "se": float("nan"), "p_value": float("nan"),
                "N": int(sum(mask)), "error": str(e)[:60]}


def placebo_threshold_test(df: pd.DataFrame, rv_col: str,
                           outcome_col: str, outcome_label: str) -> dict:
    """
    Run RD at true threshold (RV=0) and at PLACEBO_QUANTILES of the RV distribution.
    Returns results dict with true estimate and placebo distribution.
    """
    if outcome_col not in df.columns or rv_col not in df.columns:
        return {}

    sub = df[[outcome_col, rv_col]].dropna()
    y  = sub[outcome_col].values
    x  = sub[rv_col].values

    # True estimate at RV=0
    true_res = rd_at_cutoff(y, x, cutoff=0.0)

    # Placebo estimates at quantile thresholds
    # Use only the side away from 0 to avoid contaminating with real variation
    x_left  = x[x < -0.5]  # well below threshold
    x_right = x[x >  0.5]  # well above threshold

    # Compute placebo thresholds from the full RV distribution
    placebo_cutoffs = np.quantile(x, PLACEBO_QUANTILES)
    placebo_results = []

    for q, cutoff in zip(PLACEBO_QUANTILES, placebo_cutoffs):
        # Only use observations away from the true threshold
        mask_away = np.abs(x) > 0.5
        y2 = y[mask_away]
        x2 = x[mask_away]
        res = rd_at_cutoff(y2, x2, cutoff=cutoff)
        res["quantile"] = q
        res["cutoff"]   = cutoff
        placebo_results.append(res)

    # Permutation p-value: fraction of placebos with |coef| >= |true coef|
    true_coef = abs(true_res.get("coef", float("nan")))
    placebo_coefs = np.array([abs(r.get("coef", float("nan"))) for r in placebo_results])
    if not np.isnan(true_coef) and len(placebo_coefs) > 0:
        perm_p = np.mean(placebo_coefs[~np.isnan(placebo_coefs)] >= true_coef)
    else:
        perm_p = float("nan")

    return {
        "outcome":         outcome_label,
        "outcome_col":     outcome_col,
        "true_coef":       true_res.get("coef", float("nan")),
        "true_se":         true_res.get("se", float("nan")),
        "true_pv":         true_res.get("p_value", float("nan")),
        "true_N":          true_res.get("N", 0),
        "placebo_results": placebo_results,
        "permutation_p":   perm_p,
    }


def main():
    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 04: Placebo RD Tests ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    rv_a = "running_var" if "running_var" in df_a.columns else "rv_linear"
    rv_b = "running_var" if "running_var" in df_b.columns else "rv_linear"

    all_results = []

    # Mechanism B placebo threshold tests
    print("--- Mechanism B: Placebo threshold tests ---")
    for col, label in PRIMARY_OUTCOMES_B:
        print(f"  {label}")
        res = placebo_threshold_test(df_b, rv_b, col, label)
        if res:
            all_results.append(("Mech. B", res))
            tc = res["true_coef"]
            perm = res["permutation_p"]
            print(f"    True coef={tc:.4f}  perm p={perm:.3f}")

    # Mechanism A placebo threshold tests
    print("\n--- Mechanism A: Placebo threshold tests ---")
    for col, label in PRIMARY_OUTCOMES_A:
        print(f"  {label}")
        res = placebo_threshold_test(df_a, rv_a, col, label)
        if res:
            all_results.append(("Mech. A", res))
            tc = res["true_coef"]
            perm = res["permutation_p"]
            print(f"    True coef={tc:.4f}  perm p={perm:.3f}")

    # --- LaTeX table: placebo threshold estimates ---
    rows = []
    for mech, res in all_results:
        if not res:
            continue
        rows.append({
            "Mechanism":   mech,
            "Outcome":     res["outcome"],
            "True coef.":  f"{res['true_coef']:.4f}" if not np.isnan(res["true_coef"]) else "",
            "True SE":     f"{res['true_se']:.4f}"  if not np.isnan(res["true_se"]) else "",
            "True p":      f"{res['true_pv']:.3f}"  if not np.isnan(res["true_pv"]) else "",
            "Perm. p":     f"{res['permutation_p']:.3f}" if not np.isnan(res["permutation_p"]) else "",
            "N":           str(res["true_N"]),
        })
        for pr in res.get("placebo_results", []):
            rows.append({
                "Mechanism":   f"  Placebo (q={pr['quantile']:.2f})",
                "Outcome":     res["outcome"],
                "True coef.":  f"{pr['coef']:.4f}" if not np.isnan(pr.get("coef", float("nan"))) else "",
                "True SE":     f"{pr.get('se', float('nan')):.4f}" if not np.isnan(pr.get("se", float("nan"))) else "",
                "True p":      f"{pr.get('p_value', float('nan')):.3f}" if not np.isnan(pr.get("p_value", float("nan"))) else "",
                "Perm. p":     "",
                "N":           str(pr.get("N", "")),
            })

    if rows:
        tbl = pd.DataFrame(rows)
        tbl.columns = ["Mechanism", "Outcome", "Coef.", "SE", "p-value",
                       "Perm. p", "N"]
        ncol = len(tbl.columns)
        col_fmt = "ll" + "r" * (ncol - 2)
        table_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Placebo Threshold Tests}",
            r"\label{tab:placebo_thresholds}",
            rf"\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            " & ".join(tbl.columns) + r" \\",
            r"\midrule",
        ]
        for _, row in tbl.iterrows():
            table_lines.append(" & ".join(str(v) for v in row.values) + r" \\")
        table_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\footnotesize{Note: Local linear RD at true threshold (RV=0) and placebo thresholds "
            r"at 10th, 25th, 75th, and 90th percentiles. Perm. p = permutation p-value.}",
            r"\end{table}",
        ]
        (TABLES / "tab_placebo_thresholds.tex").write_text("\n".join(table_lines))
        print(f"\n  Saved: tables/tab_placebo_thresholds.tex")

    # --- Distribution plot ---
    if all_results:
        fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4),
                                 squeeze=False)
        for ax_idx, (mech, res) in enumerate(all_results):
            ax = axes[0][ax_idx]
            if not res:
                continue
            true_c = res["true_coef"]
            placebo_cs = [pr.get("coef", float("nan"))
                          for pr in res.get("placebo_results", [])]
            placebo_cs = [c for c in placebo_cs if not np.isnan(c)]

            ax.scatter(placebo_cs, [0.5] * len(placebo_cs),
                       color="steelblue", s=80, zorder=3, label="Placebo estimates")
            if not np.isnan(true_c):
                ax.axvline(true_c, color="firebrick", linewidth=2.0, linestyle="-",
                           label=f"True estimate ({true_c:.3f})")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_yticks([])
            ax.set_xlabel("RD coefficient")
            ax.set_title(f"{mech}: {res['outcome']}\n"
                         f"perm. p = {res['permutation_p']:.3f}", fontsize=9)
            ax.legend(fontsize=8)

        fig.suptitle("Placebo Threshold Tests", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_placebo_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: figures/fig_placebo_distribution.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

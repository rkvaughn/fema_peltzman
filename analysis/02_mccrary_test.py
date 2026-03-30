"""
analysis/02_mccrary_test.py
Publication-quality McCrary density plots and test statistics.

Uses rpy2 + rddensity (Cattaneo, Jansson, Ma 2020) for the formal test.
Falls back to a binned histogram approach if rpy2 is unavailable.

Outputs:
  figures/fig_mccrary_a.png    — Mechanism A running variable (historical)
  figures/fig_mccrary_b.png    — Mechanism B running variable (contemporaneous)
  tables/tab_mccrary.tex       — Test statistic table

Reads:
  data/analysis/mechanism_a.parquet
  data/analysis/mechanism_b.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    HAS_RPY2 = True
except Exception:
    HAS_RPY2 = False

ROOT   = Path(__file__).resolve().parents[1]
ANAL   = ROOT / "data" / "analysis"
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# rddensity wrapper
# ---------------------------------------------------------------------------

def run_rddensity(x: np.ndarray) -> dict:
    """
    Run rddensity from R and return a dict with:
      coef, se, t, p_conventional, p_biascorrected, bw_left, bw_right,
      fitted_left, fitted_right, grid_left, grid_right
    """
    if not HAS_RPY2:
        return {}
    try:
        base_pkg      = importr("base")
        rddensity_pkg = importr("rddensity")
        rplot_pkg     = importr("rdplot") if _r_package_available("rdplot") else None

        r_x = ro.FloatVector(x.tolist())
        fit = rddensity_pkg.rddensity(X=r_x)

        test_slot = fit.rx2("test")
        coef   = float(np.array(test_slot.rx2("t_jk"))[0])
        p_conv = float(np.array(test_slot.rx2("p_jk"))[0])
        p_bc   = float(np.array(test_slot.rx2("p_jk_bc"))[0])

        bw_slot = fit.rx2("h")
        bw_l = float(np.array(bw_slot)[0]) if len(bw_slot) >= 1 else float("nan")
        bw_r = float(np.array(bw_slot)[1]) if len(bw_slot) >= 2 else float("nan")

        # Get local polynomial density estimates for plotting
        # rddensity.plot returns a list; we extract the fitted values
        plot_obj = rddensity_pkg.rddensityPlot(fit, plotGrid="es", alpha=0.1,
                                               xlabel="Running variable",
                                               ylabel="Density")
        # Extract data frame from ggplot object via ggplot_build if ggplot2 available
        # Fallback: bin the data manually below
        return {
            "coef":           coef,
            "p_conventional": p_conv,
            "p_biascorrected": p_bc,
            "bw_left":        bw_l,
            "bw_right":       bw_r,
        }
    except Exception as e:
        print(f"  WARNING: rddensity failed ({str(e)[:80]}). Using histogram fallback.")
        return {}


def _r_package_available(pkg: str) -> bool:
    try:
        base_pkg = importr("base")
        return bool(ro.r(f"requireNamespace('{pkg}', quietly=TRUE)")[0])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Binned frequency estimate (fallback / plotting backbone)
# ---------------------------------------------------------------------------

def binned_density_estimate(x: np.ndarray, n_bins: int = 40) -> tuple:
    """
    Compute local linear density estimate from binned counts.
    Returns (midpoints_left, freq_left, midpoints_right, freq_right).
    """
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, edges = np.histogram(x, bins=bins)
    widths  = np.diff(edges)
    density = counts / (len(x) * widths)
    mids    = 0.5 * (edges[:-1] + edges[1:])

    mask_l = mids < 0
    mask_r = mids >= 0
    return mids[mask_l], density[mask_l], mids[mask_r], density[mask_r]


def local_linear_fit(midpoints: np.ndarray, freq: np.ndarray,
                     side: str = "left") -> tuple:
    """
    Fit a local-linear regression using triangular kernel.
    Returns (smooth_x, smooth_y) for plotting.
    """
    if len(midpoints) < 4:
        return midpoints, freq

    # Bandwidth: roughly half the range of midpoints
    bw = (midpoints.max() - midpoints.min()) / 2 if len(midpoints) > 1 else 1.0
    # Reference point = 0 (threshold)
    ref = 0.0
    u = (midpoints - ref) / bw
    weights = np.maximum(1 - np.abs(u), 0)

    # Weighted OLS: y = a + b * x
    W = np.diag(weights)
    X = np.column_stack([np.ones_like(midpoints), midpoints])
    try:
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ freq, rcond=None)[0]
        smooth = beta[0] + beta[1] * midpoints
    except Exception:
        smooth = freq

    return midpoints, np.maximum(smooth, 0)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_mccrary(x: np.ndarray, title: str, outpath: Path,
                 p_conv: float = None, p_bc: float = None,
                 bw_l: float = None, bw_r: float = None) -> None:
    ml, fl, mr, fr = binned_density_estimate(x, n_bins=40)
    xl, yl = local_linear_fit(ml, fl, "left")
    xr, yr = local_linear_fit(mr, fr, "right")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(ml, fl, width=(ml[1] - ml[0]) * 0.85 if len(ml) > 1 else 0.1,
           color="steelblue", alpha=0.4, label="Bin density")
    ax.bar(mr, fr, width=(mr[1] - mr[0]) * 0.85 if len(mr) > 1 else 0.1,
           color="steelblue", alpha=0.4)
    ax.plot(xl, yl, color="navy",    linewidth=2.0, label="Local linear fit (left)")
    ax.plot(xr, yr, color="navy",    linewidth=2.0, label="Local linear fit (right)")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Threshold")

    # Annotation box
    annot_parts = []
    if p_conv is not None:
        annot_parts.append(f"p (conv.) = {p_conv:.3f}")
    if p_bc is not None:
        annot_parts.append(f"p (bias-corr.) = {p_bc:.3f}")
    if annot_parts:
        ax.text(0.98, 0.95, "\n".join(annot_parts), transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Running variable (TTR-adjusted per-capita damage − threshold, $/capita)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate
    seen = set()
    uniqs = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    ax.legend(*zip(*uniqs), fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p in [ANAL / "mechanism_a.parquet", ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 02: McCrary Density Test ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    rv_a = "running_var" if "running_var" in df_a.columns else "rv_linear"
    rv_b = "running_var" if "running_var" in df_b.columns else "rv_linear"

    results = []

    for name, df, rv_col, fig_out in [
        ("Mechanism A (historical RV)", df_a, rv_a, FIGS / "fig_mccrary_a.png"),
        ("Mechanism B (contemp. RV)",   df_b, rv_b, FIGS / "fig_mccrary_b.png"),
    ]:
        if rv_col not in df.columns:
            print(f"  WARNING: {rv_col} not in {name} — skipping")
            results.append({"Dataset": name, "N": len(df),
                            "p_conv": "", "p_bc": "", "bw_l": "", "bw_r": "", "Result": "SKIP"})
            continue

        x_vals = df[rv_col].dropna().values
        print(f"\n--- {name} (N={len(x_vals):,}) ---")

        p_conv, p_bc, bw_l, bw_r = None, None, None, None

        if HAS_RPY2 and len(x_vals) >= 50:
            rd_res = run_rddensity(x_vals)
            p_conv = rd_res.get("p_conventional")
            p_bc   = rd_res.get("p_biascorrected")
            bw_l   = rd_res.get("bw_left")
            bw_r   = rd_res.get("bw_right")
            if p_conv is not None:
                print(f"  rddensity: p_conventional={p_conv:.4f}, p_biascorrected={p_bc:.4f}")
                print(f"  Bandwidth: left={bw_l:.3f}, right={bw_r:.3f}")

        status = ""
        if p_conv is not None:
            status = "FAIL (p<0.05)" if p_conv < 0.05 else "PASS"
        elif not HAS_RPY2:
            status = "rpy2 unavailable"

        results.append({
            "Dataset":  name,
            "N":        len(x_vals),
            "p_conv":   f"{p_conv:.4f}" if p_conv is not None else "",
            "p_bc":     f"{p_bc:.4f}"   if p_bc is not None else "",
            "bw_l":     f"{bw_l:.3f}"   if bw_l is not None else "",
            "bw_r":     f"{bw_r:.3f}"   if bw_r is not None else "",
            "Result":   status,
        })

        plot_mccrary(
            x_vals, f"McCrary Density Test — {name}",
            fig_out, p_conv=p_conv, p_bc=p_bc, bw_l=bw_l, bw_r=bw_r,
        )

    # Summary table
    tbl = pd.DataFrame(results)
    tbl.columns = ["Dataset", "N", "p (conv.)", "p (bias-corr.)",
                   "BW left", "BW right", "Result"]

    ncol = len(tbl.columns)
    col_fmt = "l" + "r" * (ncol - 1)
    header = " & ".join(tbl.columns) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{McCrary Density Test — Running Variable Manipulation Check}",
        r"\label{tab:mccrary}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\footnotesize{Note: rddensity (Cattaneo, Jansson, and Ma 2020). "
              r"Null hypothesis: density is continuous at threshold. "
              r"p $<$ 0.05 indicates manipulation.}",
              r"\end{table}"]
    (TABLES / "tab_mccrary.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_mccrary.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

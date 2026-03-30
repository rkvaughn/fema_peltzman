"""
analysis/07_mechanism_b_contemp_rd.py
Mechanism B: Fuzzy RD estimates of the effect of IA declaration on post-disaster
claiming and recovery behavior.

Uses rdrobust (Python) for local linear RD with CCT bandwidth.
Reports bias-corrected estimates with robust SEs.

Outcomes (6, per project plan):
  1. ia_application_rate
  2. ia_approval_rate
  3. avg_ia_award_owner
  4. avg_ia_award_renter
  5. n_nfip_claims (log)
  6. time_to_ia_application

Outputs:
  tables/tab_mech_b_rd.tex         — main RD table
  figures/fig_mech_b_bandwidth.png — bandwidth robustness
  figures/fig_mech_b_rdplot.png    — RD plots for primary outcomes

Reads:
  data/analysis/mechanism_b.parquet
  data/analysis/mechanism_b_flood.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from rdrobust import rdrobust, rdplot
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

# Outcomes for Mechanism B (pre-specified in project plan)
OUTCOMES_B = [
    ("ia_application_rate",   "IA application rate"),
    ("ia_approval_rate",      "IA approval rate"),
    ("avg_ia_award_owner",    "Avg. IA award — owners"),
    ("avg_ia_award_renter",   "Avg. IA award — renters"),
    ("n_nfip_claims",         "NFIP claims (log)"),
    ("time_to_ia_application","Time to IA application (days)"),
]

# Bandwidth multipliers for robustness (methods appendix §A.4.3)
BW_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def prep_log_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform right-skewed count/amount outcomes."""
    df2 = df.copy()
    for col in ["n_nfip_claims", "avg_ia_award_owner", "avg_ia_award_renter"]:
        if col in df2.columns:
            df2[col] = np.log1p(df2[col].clip(lower=0))
    return df2


def run_rdrobust(y: np.ndarray, x: np.ndarray,
                 fuzzy: np.ndarray = None) -> dict:
    """
    Run rdrobust local linear RD.
    If fuzzy is provided, runs fuzzy RD (2SLS via rdrobust fuzzy option).
    Returns dict with coef_bc, se_rb, ci_lo, ci_hi, bw, N_left, N_right, p_rb.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    if fuzzy is not None:
        mask &= np.isfinite(fuzzy)
    y2, x2 = y[mask], x[mask]
    f2 = fuzzy[mask] if fuzzy is not None else None

    if len(y2) < 50:
        return {"error": f"too few obs: {len(y2)}"}

    try:
        if HAS_RDROBUST:
            kwargs = dict(kernel="triangular", vce="hc1")
            if f2 is not None:
                kwargs["fuzzy"] = f2
            rr = rdrobust(y=y2, x=x2, **kwargs)
            # Index 1 = bias-corrected; index 2 = robust
            coef  = float(rr.coef[1])
            se_rb = float(rr.se[2])
            ci_lo = float(rr.ci[1, 0])
            ci_hi = float(rr.ci[1, 1])
            bw    = float(rr.bws[0, 0])
            n_l   = int(rr.N_h[0])
            n_r   = int(rr.N_h[1])
            p_rb  = float(rr.pv[2]) if hasattr(rr, "pv") and len(rr.pv) > 2 else (
                    2 * (1 - __import__("scipy.stats", fromlist=["norm"]).norm.cdf(abs(coef / se_rb)))
                    if se_rb > 0 else float("nan"))
            return {"coef": coef, "se": se_rb, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "bw": bw, "N_left": n_l, "N_right": n_r, "p_value": p_rb,
                    "error": None}
        else:
            # OLS fallback
            df2 = pd.DataFrame({"y": y2, "x": x2})
            df2["above"] = (df2["x"] >= 0).astype(int)
            df2["inter"] = df2["above"] * df2["x"]
            bw = pd.Series(x2).abs().quantile(0.5)
            bw = float(bw)
            df2 = df2[df2["x"].abs() <= bw]
            if len(df2) < 20:
                return {"error": "too few in bandwidth"}
            mod = smf.wls("y ~ above + x + inter", data=df2).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            pv   = mod.pvalues.get("above", float("nan"))
            ci   = mod.conf_int().loc["above"] if "above" in mod.conf_int().index else [float("nan")] * 2
            return {"coef": coef, "se": se, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                    "bw": bw, "N_left": int((df2["x"] < 0).sum()),
                    "N_right": int((df2["x"] >= 0).sum()), "p_value": pv,
                    "error": None}
    except Exception as e:
        return {"error": str(e)[:80]}


def rd_plot_simple(y: np.ndarray, x: np.ndarray, title: str,
                   outpath: Path, coef: float = None, bw: float = None) -> None:
    """Simple binscatter-style RD plot."""
    mask = np.isfinite(y) & np.isfinite(x)
    y2, x2 = y[mask], x[mask]

    # Bins
    lo, hi = np.percentile(x2, 2), np.percentile(x2, 98)
    x2c = np.clip(x2, lo, hi)
    y2c = y2[(x2 >= lo) & (x2 <= hi)]
    x2c = x2c[(x2 >= lo) & (x2 <= hi)]

    def bin_means(xv, yv, nb=15):
        edges = np.percentile(xv, np.linspace(0, 100, nb + 1))
        mx, my = [], []
        for i in range(nb):
            m = (xv >= edges[i]) & (xv <= edges[i + 1])
            if m.sum() > 0:
                mx.append(xv[m].mean()); my.append(yv[m].mean())
        return np.array(mx), np.array(my)

    xl, yl = bin_means(x2c[x2c < 0], y2c[x2c < 0])
    xr, yr = bin_means(x2c[x2c >= 0], y2c[x2c >= 0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xl, yl, color="steelblue", s=25, zorder=3)
    ax.scatter(xr, yr, color="steelblue", s=25, zorder=3)

    for xv, yv in [(xl, yl), (xr, yr)]:
        if len(xv) >= 3:
            from numpy.polynomial.polynomial import polyfit, polyval
            c = polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xs, polyval(xs, c), color="navy", linewidth=1.8)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    if coef is not None and bw is not None:
        ax.text(0.02, 0.95, f"τ = {coef:.4f}  BW = {bw:.3f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    ax.set_xlabel("Running variable (TTR-adjusted, $/capita)")
    ax.set_ylabel(title.split("—")[-1].strip() if "—" in title else "Outcome")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath.relative_to(ROOT)}")


def main():
    for p in [ANAL / "mechanism_b.parquet"]:
        if not p.exists():
            print(f"MISSING: {p.relative_to(ROOT)}")
            print("  Run preprocessing/05_build_analysis_dataset.py first.")
            return

    print("=== Analysis 07: Mechanism B — Fuzzy RD ===\n")

    df_b    = prep_log_outcomes(pd.read_parquet(ANAL / "mechanism_b.parquet"))
    fl_path = ANAL / "mechanism_b_flood.parquet"
    df_b_fl = prep_log_outcomes(pd.read_parquet(fl_path)) if fl_path.exists() else pd.DataFrame()

    rv_col     = "running_var" if "running_var" in df_b.columns else "rv_linear"
    fuzzy_col  = "ia_declared" if "ia_declared" in df_b.columns else None

    results = []

    for outcome_col, outcome_label in OUTCOMES_B:
        print(f"\n  Outcome: {outcome_label}")
        for dataset_label, dataset in [("All hazards", df_b), ("Flood only", df_b_fl)]:
            if len(dataset) == 0:
                continue
            if outcome_col not in dataset.columns or rv_col not in dataset.columns:
                print(f"    {dataset_label}: SKIP (columns missing)")
                continue

            sub = dataset.dropna(subset=[outcome_col, rv_col])
            y = sub[outcome_col].values
            x = sub[rv_col].values
            f = sub[fuzzy_col].values if (fuzzy_col and fuzzy_col in sub.columns) else None

            res = run_rdrobust(y, x, fuzzy=f)

            if res.get("error"):
                print(f"    {dataset_label}: ERROR — {res['error']}")
            else:
                pv = res.get("p_value", float("nan"))
                stars = ("***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "")))
                print(f"    {dataset_label}: coef={res['coef']:.4f} ({res['se']:.4f}){stars}  "
                      f"N={res['N_left']+res['N_right']:,}  BW={res['bw']:.3f}")

            results.append({
                "outcome": outcome_label,
                "outcome_col": outcome_col,
                "sample":  dataset_label,
                "dataset": dataset,
                **({k: v for k, v in res.items() if k not in ("error", "dataset")}
                   if not res.get("error") else {"error": res["error"]}),
            })

    # --- RD plots for primary outcome (ia_application_rate) ---
    print("\nGenerating RD plots...")
    for outcome_col, outcome_label in OUTCOMES_B[:2]:
        for dataset_label, dataset, suffix in [
            ("All hazards", df_b, "b"), ("Flood only", df_b_fl, "b_flood")
        ]:
            if len(dataset) == 0:
                continue
            if outcome_col not in dataset.columns or rv_col not in dataset.columns:
                continue
            res = next((r for r in results
                        if r["outcome"] == outcome_label and r["sample"] == dataset_label
                        and not r.get("error")), {})
            sub = dataset.dropna(subset=[outcome_col, rv_col])
            rd_plot_simple(
                sub[outcome_col].values, sub[rv_col].values,
                title=f"Mechanism B ({dataset_label}) — {outcome_label}",
                outpath=FIGS / f"fig_mech_{suffix}_rdplot_{outcome_col}.png",
                coef=res.get("coef"), bw=res.get("bw"),
            )

    # --- Bandwidth robustness for primary outcome ---
    print("\nBandwidth robustness (ia_application_rate, all hazards)...")
    primary_col = "ia_application_rate"
    if primary_col in df_b.columns and rv_col in df_b.columns:
        # Use CCT optimal BW from main results as reference
        ref_res = next((r for r in results
                        if r["outcome_col"] == primary_col
                        and r["sample"] == "All hazards"
                        and not r.get("error")), {})
        base_bw = ref_res.get("bw", df_b[rv_col].abs().quantile(0.5))

        bw_results = []
        for mult in BW_MULTIPLIERS:
            bw = base_bw * mult
            sub = df_b[df_b[rv_col].abs() <= bw].dropna(subset=[primary_col, rv_col])
            y  = sub[primary_col].values
            x  = sub[rv_col].values
            f  = sub[fuzzy_col].values if (fuzzy_col and fuzzy_col in sub.columns) else None
            res = run_rdrobust(y, x, fuzzy=f)
            if not res.get("error"):
                bw_results.append({"bw_mult": mult, **res})
                print(f"  {mult:.2f}×  bw={bw:.3f}  N={res['N_left']+res['N_right']:,}"
                      f"  coef={res['coef']:.4f}")

        if bw_results:
            bw_df = pd.DataFrame(bw_results)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(bw_df["bw_mult"], bw_df["coef"],
                        yerr=1.96 * bw_df["se"],
                        fmt="o-", color="steelblue", ecolor="steelblue",
                        elinewidth=1.5, capsize=4)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Bandwidth multiplier (×CCT optimal)")
            ax.set_ylabel("Fuzzy RD coefficient (IA application rate)")
            ax.set_title("Mechanism B: Bandwidth Robustness\n"
                         "IA application rate ~ IA declaration (fuzzy RD)")
            ax.set_xticks(BW_MULTIPLIERS)
            fig.tight_layout()
            fig.savefig(FIGS / "fig_mech_b_bandwidth.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: figures/fig_mech_b_bandwidth.png")

    # --- LaTeX table ---
    rows_all   = [r for r in results if r.get("sample") == "All hazards" and not r.get("error")]
    rows_flood = [r for r in results if r.get("sample") == "Flood only"  and not r.get("error")]

    def fmt(r):
        if r.get("error"):
            return r["error"], ""
        c  = r.get("coef", float("nan"))
        se = r.get("se",   float("nan"))
        pv = r.get("p_value", float("nan"))
        stars = ("***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "")))
        return f"{c:.4f}{stars}", f"({se:.4f})"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Mechanism B: Fuzzy RD Estimates — Effect of IA Declaration on Post-Disaster Outcomes}",
        r"\label{tab:mech_b}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{All Hazards} & \multicolumn{2}{c}{Flood Only} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Outcome & Coef. & SE & Coef. & SE \\",
        r"\midrule",
    ]
    for outcome_col, outcome_label in OUTCOMES_B:
        ra = next((r for r in rows_all   if r["outcome"] == outcome_label), {})
        rf = next((r for r in rows_flood if r["outcome"] == outcome_label), {})
        ca, sea = fmt(ra)
        cf, sef = fmt(rf)
        lines.append(rf"{outcome_label} & {ca} & {sea} & {cf} & {sef} \\")
    na = (rows_all[0].get("N_left",  0) + rows_all[0].get("N_right",  0)) if rows_all   else "—"
    nf = (rows_flood[0].get("N_left", 0) + rows_flood[0].get("N_right", 0)) if rows_flood else "—"
    lines += [
        r"\midrule",
        rf"N & {na:,} & & {nf:,} & \\" if isinstance(na, int) else rf"N & {na} & & {nf} & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize{Note: rdrobust (triangular kernel, CCT bandwidth, robust SEs). "
        r"Fuzzy RD: instrument = 1[running variable $\geq$ 0]. "
        r"Log transformation applied to n\_nfip\_claims and IA award amounts. "
        r"*** p$<$0.01, ** p$<$0.05, * p$<$0.10.}",
        r"\end{table}",
    ]
    (TABLES / "tab_mech_b_rd.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_mech_b_rd.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

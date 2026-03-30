"""
analysis/10_heterogeneity.py
Heterogeneity analysis across key dimensions.

(1) Tenure interaction — owner-occupancy moderates the Peltzman effect
    (methods appendix §G.1: split by tercile of pct_owner_occ)

(2) Income terciles — low/medium/high median household income

(3) Temporal decay curve — treatment effect by years-since-prior bin
    (bins: [0,2], (2,5], (5,9], (9,15] per project plan / Gallagher 2014)

(4) Pre/post NFIP reform split — BW2012 Biggert-Waters Act (2012) as structural break

Outputs:
  tables/tab_heterogeneity.tex          — heterogeneity table
  figures/fig_tenure_interaction.png    — tenure interaction plot
  figures/fig_temporal_decay.png        — temporal decay curve
  figures/fig_income_terciles.png       — income tercile estimates

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

# Temporal decay bins (pre-specified in project plan §temporal decay / methods appendix §G.2)
TEMPORAL_BINS   = [0, 2, 5, 9, 15, float("inf")]
TEMPORAL_LABELS = ["0–2yr", "3–5yr", "6–9yr", "10–15yr", "15+yr"]

# NFIP reform structural break year (Biggert-Waters 2012)
# Pre-specified in methods appendix §G.4 as a sensitivity check.
BW_REFORM_YEAR = 2012

PRIMARY_OUTCOME_A = "contents_damage_ratio"
PRIMARY_OUTCOME_B = "ia_application_rate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_rd(df: pd.DataFrame, outcome: str, rv_col: str,
              fuzzy_col: str = None, label: str = "") -> dict:
    if outcome not in df.columns or rv_col not in df.columns:
        return {"error": "missing columns", "label": label}

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
        return {"error": f"N={len(y)} < 30", "label": label}

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
                    "N": int(sum(mask)), "error": None, "label": label}
        else:
            df2 = pd.DataFrame({"y": y, "x": x})
            df2["above"] = (df2["x"] >= 0).astype(int)
            df2["inter"] = df2["above"] * df2["x"]
            bw = float(pd.Series(x).abs().quantile(0.5))
            df2 = df2[df2["x"].abs() <= bw]
            if len(df2) < 20:
                return {"error": "too few in bw", "label": label}
            mod = smf.wls("y ~ above + x + inter", data=df2).fit()
            coef = mod.params.get("above", float("nan"))
            se   = mod.bse.get("above", float("nan"))
            pv   = mod.pvalues.get("above", float("nan"))
            return {"coef": coef, "se": se, "pv": pv, "bw": bw,
                    "N": len(df2), "error": None, "label": label}
    except Exception as e:
        return {"error": str(e)[:60], "label": label}


def fmt_coef(r: dict) -> str:
    if r.get("error"):
        return r["error"]
    c  = r.get("coef", float("nan"))
    pv = r.get("pv", float("nan"))
    se = r.get("se", float("nan"))
    stars = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
    return f"{c:.4f}{stars} ({se:.4f})"


def coef_plot(results: list, xlabel: str, title: str, outpath: Path) -> None:
    valid = [r for r in results if not r.get("error") and "coef" in r]
    if not valid:
        print(f"  No valid estimates for {title} — skipping plot")
        return
    labels = [r["label"] for r in valid]
    coefs  = [r["coef"] for r in valid]
    ses    = [r["se"]   for r in valid]
    ys = range(len(valid))
    fig, ax = plt.subplots(figsize=(8, max(3, len(valid) * 0.6 + 1)))
    ax.errorbar(coefs, list(ys), xerr=[1.96 * s for s in ses],
                fmt="o", color="steelblue", ecolor="steelblue",
                elinewidth=1.5, capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# (1) Tenure interaction
# ---------------------------------------------------------------------------

def tenure_interaction(df: pd.DataFrame, outcome: str, rv_col: str,
                       fuzzy_col: str = None) -> list:
    """
    Split by tercile of pct_owner_occ.
    Returns one estimate per tercile.
    """
    if "pct_owner_occ" not in df.columns:
        print("  SKIP tenure: pct_owner_occ not found")
        return []

    df2 = df.copy()
    df2["occ_tercile"] = pd.qcut(df2["pct_owner_occ"], 3,
                                 labels=["Low ownership", "Mid ownership", "High ownership"])
    results = []
    for label in ["Low ownership", "Mid ownership", "High ownership"]:
        sub = df2[df2["occ_tercile"] == label]
        res = simple_rd(sub, outcome, rv_col, fuzzy_col, label=label)
        results.append(res)
        if not res.get("error"):
            print(f"  {label}: coef={res['coef']:.4f}  se={res['se']:.4f}  N={res['N']:,}")
        else:
            print(f"  {label}: ERROR — {res['error']}")
    return results


# ---------------------------------------------------------------------------
# (2) Income terciles
# ---------------------------------------------------------------------------

def income_terciles(df: pd.DataFrame, outcome: str, rv_col: str,
                    fuzzy_col: str = None) -> list:
    inc_col = "median_household_income"
    if inc_col not in df.columns:
        print("  SKIP income: median_household_income not found")
        return []

    df2 = df.copy()
    df2["inc_tercile"] = pd.qcut(df2[inc_col], 3,
                                 labels=["Low income", "Mid income", "High income"])
    results = []
    for label in ["Low income", "Mid income", "High income"]:
        sub = df2[df2["inc_tercile"] == label]
        res = simple_rd(sub, outcome, rv_col, fuzzy_col, label=label)
        results.append(res)
        if not res.get("error"):
            print(f"  {label}: coef={res['coef']:.4f}  se={res['se']:.4f}  N={res['N']:,}")
        else:
            print(f"  {label}: ERROR — {res['error']}")
    return results


# ---------------------------------------------------------------------------
# (3) Temporal decay curve (Mechanism A only)
# ---------------------------------------------------------------------------

def temporal_decay(df: pd.DataFrame, outcome: str, rv_col: str,
                   fuzzy_col: str = None) -> list:
    """
    Estimate RD separately for each years-since-prior bin.
    Bins pre-specified in project plan / methods appendix §G.2.
    """
    if "years_since_prior" not in df.columns:
        print("  SKIP temporal decay: years_since_prior not found")
        return []

    df2 = df.copy()
    df2["bin"] = pd.cut(df2["years_since_prior"],
                        bins=TEMPORAL_BINS, labels=TEMPORAL_LABELS, right=True)
    results = []
    for lbl in TEMPORAL_LABELS:
        sub = df2[df2["bin"] == lbl]
        res = simple_rd(sub, outcome, rv_col, fuzzy_col, label=lbl)
        results.append(res)
        if not res.get("error"):
            print(f"  {lbl}: coef={res['coef']:.4f}  se={res['se']:.4f}  N={res['N']:,}")
        else:
            print(f"  {lbl}: ERROR — {res['error']}")
    return results


# ---------------------------------------------------------------------------
# (4) Pre/post NFIP reform
# ---------------------------------------------------------------------------

def reform_split(df: pd.DataFrame, outcome: str, rv_col: str,
                 fuzzy_col: str = None) -> list:
    """
    Split by event_year relative to BW_REFORM_YEAR (2012 Biggert-Waters Act).
    Pre-specified in methods appendix §G.4.
    """
    if "event_year" not in df.columns:
        print("  SKIP reform split: event_year not found")
        return []
    results = []
    for label, mask_fn in [
        (f"Pre-{BW_REFORM_YEAR}",  lambda d: d["event_year"] <  BW_REFORM_YEAR),
        (f"Post-{BW_REFORM_YEAR}", lambda d: d["event_year"] >= BW_REFORM_YEAR),
    ]:
        sub = df[mask_fn(df)]
        res = simple_rd(sub, outcome, rv_col, fuzzy_col, label=label)
        results.append(res)
        if not res.get("error"):
            print(f"  {label}: coef={res['coef']:.4f}  se={res['se']:.4f}  N={res['N']:,}")
        else:
            print(f"  {label}: ERROR — {res['error']}")
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

    print("=== Analysis 10: Heterogeneity ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    rv_a     = "running_var"         if "running_var"         in df_a.columns else "rv_linear"
    rv_b     = "running_var"         if "running_var"         in df_b.columns else "rv_linear"
    prior_rv = "prior_running_var_raw" if "prior_running_var_raw" in df_a.columns else rv_a
    fuzzy_a  = "prior_ia_received"  if "prior_ia_received"   in df_a.columns else None
    fuzzy_b  = "ia_declared"        if "ia_declared"         in df_b.columns else None

    table_sections = []

    # -----------------------------------------------------------------------
    # (1) Tenure interaction
    # -----------------------------------------------------------------------
    print("--- (1) Tenure interaction (Mechanism A) ---")
    tenure_a = tenure_interaction(df_a, PRIMARY_OUTCOME_A, prior_rv, fuzzy_a)
    if tenure_a:
        coef_plot(tenure_a, "Estimate (contents damage ratio)",
                  "Mechanism A: Tenure Interaction\n(split by owner-occupancy tercile)",
                  FIGS / "fig_tenure_interaction.png")
        table_sections.append(("A", "Tenure (owner-occ. tercile)", tenure_a))

    print("\n--- (1b) Tenure interaction (Mechanism B) ---")
    tenure_b = tenure_interaction(df_b, PRIMARY_OUTCOME_B, rv_b, fuzzy_b)
    if tenure_b:
        table_sections.append(("B", "Tenure (owner-occ. tercile)", tenure_b))

    # -----------------------------------------------------------------------
    # (2) Income terciles
    # -----------------------------------------------------------------------
    print("\n--- (2) Income terciles (Mechanism A) ---")
    income_a = income_terciles(df_a, PRIMARY_OUTCOME_A, prior_rv, fuzzy_a)
    if income_a:
        coef_plot(income_a, "Estimate (contents damage ratio)",
                  "Mechanism A: Income Tercile Heterogeneity",
                  FIGS / "fig_income_terciles.png")
        table_sections.append(("A", "Income tercile", income_a))

    print("\n--- (2b) Income terciles (Mechanism B) ---")
    income_b = income_terciles(df_b, PRIMARY_OUTCOME_B, rv_b, fuzzy_b)
    if income_b:
        table_sections.append(("B", "Income tercile", income_b))

    # -----------------------------------------------------------------------
    # (3) Temporal decay (Mechanism A only)
    # -----------------------------------------------------------------------
    print("\n--- (3) Temporal decay curve (Mechanism A) ---")
    decay_a = temporal_decay(df_a, PRIMARY_OUTCOME_A, prior_rv, fuzzy_a)
    if decay_a:
        table_sections.append(("A", "Years since prior event", decay_a))

        # Line plot connecting bin midpoints — illustrates attenuation
        valid_decay = [r for r in decay_a if not r.get("error") and "coef" in r]
        if valid_decay:
            fig, ax = plt.subplots(figsize=(8, 5))
            xs    = range(len(valid_decay))
            coefs = [r["coef"] for r in valid_decay]
            ses   = [r["se"]   for r in valid_decay]
            labels_d = [r["label"] for r in valid_decay]

            ax.errorbar(xs, coefs, yerr=[1.96 * s for s in ses],
                        fmt="o-", color="steelblue", ecolor="steelblue",
                        elinewidth=1.5, capsize=5, markersize=8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(labels_d)
            ax.set_xlabel("Years since prior IA event")
            ax.set_ylabel("IV/2SLS coefficient (contents damage ratio)")
            ax.set_title("Mechanism A: Temporal Decay of Peltzman Effect\n"
                         "(bins pre-specified per Gallagher 2014)")
            fig.tight_layout()
            fig.savefig(FIGS / "fig_temporal_decay.png", dpi=150)
            plt.close(fig)
            print(f"\n  Saved: figures/fig_temporal_decay.png")

    # -----------------------------------------------------------------------
    # (4) Pre/post NFIP reform
    # -----------------------------------------------------------------------
    print("\n--- (4) Pre/post NFIP reform split ---")
    print("  Mechanism A:")
    reform_a = reform_split(df_a, PRIMARY_OUTCOME_A, prior_rv, fuzzy_a)
    if reform_a:
        table_sections.append(("A", f"Pre/post BW Act ({BW_REFORM_YEAR})", reform_a))

    print("  Mechanism B:")
    reform_b = reform_split(df_b, PRIMARY_OUTCOME_B, rv_b, fuzzy_b)
    if reform_b:
        table_sections.append(("B", f"Pre/post BW Act ({BW_REFORM_YEAR})", reform_b))

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Heterogeneity Analysis}",
        r"\label{tab:heterogeneity}",
        r"\begin{tabular}{llllr}",
        r"\toprule",
        r"Mechanism & Dimension & Subgroup & Estimate & N \\",
        r"\midrule",
    ]
    for mech, dim, res_list in table_sections:
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{Mechanism {mech} — {dim}}}}} \\"
        )
        for r in res_list:
            lbl = r.get("label", "")
            est = fmt_coef(r)
            n   = f"{r.get('N', ''):,}" if isinstance(r.get("N"), int) else str(r.get("N", ""))
            lines.append(f"  & & {lbl} & {est} & {n} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize{Note: Each row is an independent RD or IV estimate on the indicated subgroup. "
        r"Tenure and income splits at tercile boundaries of the respective ACS variable. "
        r"Temporal decay bins pre-specified per Gallagher (2014). "
        r"BW Act = Biggert-Waters Flood Insurance Reform Act (2012). "
        r"*** p$<$0.01, ** p$<$0.05, * p$<$0.10.}",
        r"\end{table}",
    ]
    (TABLES / "tab_heterogeneity.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_heterogeneity.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

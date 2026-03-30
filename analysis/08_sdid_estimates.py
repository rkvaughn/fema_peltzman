"""
analysis/08_sdid_estimates.py
Synthetic Difference-in-Differences (SDID) via rpy2 + synthdid R package.

Panel construction:
  - Unit: zcta5 (ZIP)
  - Time: event_period (year of disaster event)
  - Treatment: prior_ia_received (Mechanism A) or ia_declared (Mechanism B)
  - Outcome: contents_damage_ratio (primary); ia_application_rate (secondary)

Pre-processing:
  - Restrict to ZIPs with >= 80% period coverage (pre-specified in project plan)
  - Impute missing outcomes with 0 (documented in code comment)
  - Balanced panel required by synthdid::panel.matrices()

Inference:
  - Permutation SEs: K=1000 random reassignments (methods appendix §I.2)
  - Placebo variance estimator as alternative

Outputs:
  tables/tab_sdid.tex           — SDID results table
  figures/fig_sdid_mech_a.png   — SDID synthetic control path plot (Mechanism A)
  figures/fig_sdid_mech_b.png   — SDID synthetic control path plot (Mechanism B)

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

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
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

# Minimum period coverage fraction for balanced panel (pre-specified in project plan)
MIN_COVERAGE = 0.80

# Permutation iterations (methods appendix §I.2, pre-specified K=1000)
N_PERMUTATIONS = 1000

# SDID outcomes by mechanism
MECH_A_OUTCOME = "contents_damage_ratio"
MECH_B_OUTCOME = "ia_application_rate"
MECH_A_TREAT   = "prior_ia_received"
MECH_B_TREAT   = "ia_declared"


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_balanced_panel(df: pd.DataFrame,
                         unit_col: str, time_col: str,
                         outcome_col: str, treat_col: str,
                         min_coverage: float = MIN_COVERAGE) -> pd.DataFrame | None:
    """
    Build a balanced panel from the analysis dataset.
    Restricts to units with >= min_coverage fraction of time periods present.
    Missing outcomes imputed to 0 (documented: assumes missing = no claim activity).

    Returns long-format DataFrame or None if insufficient data.
    """
    needed = [unit_col, time_col, outcome_col, treat_col]
    available = [c for c in needed if c in df.columns]
    if len(available) < len(needed):
        missing = [c for c in needed if c not in df.columns]
        print(f"  Panel construction: missing columns {missing}")
        return None

    sub = df[[unit_col, time_col, outcome_col, treat_col]].copy()
    sub[time_col] = sub[time_col].astype(int)
    all_periods = sorted(sub[time_col].unique())
    n_periods = len(all_periods)

    # Count periods per unit
    unit_counts = sub.groupby(unit_col)[time_col].nunique()
    min_periods = int(np.ceil(min_coverage * n_periods))
    qualifying_units = unit_counts[unit_counts >= min_periods].index
    sub = sub[sub[unit_col].isin(qualifying_units)]

    if len(sub) == 0:
        print(f"  No units with >= {min_coverage:.0%} coverage")
        return None

    # Build full grid and impute missing with 0
    # Imputation note: missing outcome = no NFIP/IA claim activity in that period;
    # imputing 0 is conservative and documented per project plan.
    grid = pd.MultiIndex.from_product(
        [qualifying_units, all_periods], names=[unit_col, time_col]
    ).to_frame(index=False)
    panel = grid.merge(sub, on=[unit_col, time_col], how="left")
    panel[outcome_col] = panel[outcome_col].fillna(0)  # impute missing = 0 (see note above)

    # Treatment: carry forward (a unit stays treated once treated)
    panel = panel.sort_values([unit_col, time_col])
    panel[treat_col] = panel.groupby(unit_col)[treat_col].transform(
        lambda s: s.ffill().fillna(0)
    )

    n_units   = panel[unit_col].nunique()
    n_treated = panel.groupby(unit_col)[treat_col].max().gt(0).sum()
    print(f"  Balanced panel: {n_units:,} units × {n_periods} periods  "
          f"({n_treated:,} treated)")

    return panel


# ---------------------------------------------------------------------------
# SDID via rpy2
# ---------------------------------------------------------------------------

def run_sdid_r(panel: pd.DataFrame, unit_col: str, time_col: str,
               outcome_col: str, treat_col: str,
               n_perm: int = N_PERMUTATIONS) -> dict:
    """
    Call synthdid::synthdid_estimate() via rpy2.
    Returns dict with tau_hat, se_placebo, se_permutation, p_value.
    """
    if not HAS_RPY2:
        return {"error": "rpy2 not available"}

    try:
        synthdid_pkg = importr("synthdid")
        base_pkg     = importr("base")

        # Convert to R data frame
        r_df = pandas2ri.py2rpy(panel[[unit_col, time_col, outcome_col, treat_col]])

        # panel.matrices() requires unit, time, outcome, treatment columns
        panel_mats = synthdid_pkg.panel_dot_matrices(
            r_df,
            unit      = unit_col,
            time      = time_col,
            outcome   = outcome_col,
            treatment = treat_col,
        )

        Y  = panel_mats.rx2("Y")
        N0 = panel_mats.rx2("N0")
        T0 = panel_mats.rx2("T0")

        tau_hat = synthdid_pkg.synthdid_estimate(Y, N0, T0)
        tau_val = float(base_pkg.c(tau_hat)[0])

        # Placebo SE (leave-one-out)
        try:
            se_vec    = synthdid_pkg.sqrt(synthdid_pkg.vcov(tau_hat, method="placebo"))
            se_placebo = float(base_pkg.c(se_vec)[0])
        except Exception:
            se_placebo = float("nan")

        # Permutation SE (K = n_perm random reassignments from control pool)
        # Uses synthdid::vcov with method="bootstrap" as proxy when permutation
        # is not directly available in the R API
        try:
            se_vec_b   = synthdid_pkg.sqrt(synthdid_pkg.vcov(tau_hat, method="bootstrap"))
            se_perm    = float(base_pkg.c(se_vec_b)[0])
        except Exception:
            se_perm = float("nan")

        # p-value from placebo SE
        from scipy.stats import norm
        pv = 2 * (1 - norm.cdf(abs(tau_val) / se_placebo)) if not np.isnan(se_placebo) and se_placebo > 0 else float("nan")

        return {
            "tau":          tau_val,
            "se_placebo":   se_placebo,
            "se_bootstrap": se_perm,
            "p_value":      pv,
            "error":        None,
        }

    except Exception as e:
        return {"error": str(e)[:100]}


def run_sdid_permutation(panel: pd.DataFrame, unit_col: str, time_col: str,
                         outcome_col: str, treat_col: str,
                         tau_obs: float, n_perm: int = N_PERMUTATIONS) -> float:
    """
    Permutation inference: K random reassignments from control pool.
    Returns permutation p-value (two-sided).
    Pre-specified: K=1000 (methods appendix §I.2).
    """
    if not HAS_RPY2:
        return float("nan")

    try:
        synthdid_pkg = importr("synthdid")
        base_pkg     = importr("base")

        n_treated = int(panel.groupby(unit_col)[treat_col].max().gt(0).sum())
        control_units = panel.groupby(unit_col)[treat_col].max()
        control_units = control_units[control_units == 0].index.tolist()

        if len(control_units) < n_treated:
            return float("nan")

        rng = np.random.default_rng(seed=42)
        placebo_taus = []
        for _ in range(n_perm):
            placebo_treated = rng.choice(control_units, size=n_treated, replace=False)
            panel_perm = panel.copy()
            panel_perm[treat_col] = panel_perm[unit_col].isin(placebo_treated).astype(float)

            r_df_p = pandas2ri.py2rpy(panel_perm[[unit_col, time_col, outcome_col, treat_col]])
            try:
                pm = synthdid_pkg.panel_dot_matrices(
                    r_df_p, unit=unit_col, time=time_col,
                    outcome=outcome_col, treatment=treat_col,
                )
                tau_p = synthdid_pkg.synthdid_estimate(pm.rx2("Y"), pm.rx2("N0"), pm.rx2("T0"))
                placebo_taus.append(float(base_pkg.c(tau_p)[0]))
            except Exception:
                pass

        if not placebo_taus:
            return float("nan")

        perm_p = np.mean(np.abs(placebo_taus) >= abs(tau_obs))
        return float(perm_p)

    except Exception as e:
        print(f"  Permutation inference error: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Plot: synthetic control path
# ---------------------------------------------------------------------------

def plot_sdid_path(panel: pd.DataFrame, unit_col: str, time_col: str,
                   outcome_col: str, treat_col: str,
                   tau: float, title: str, outpath: Path) -> None:
    treated_units  = panel.groupby(unit_col)[treat_col].max()
    treated_units  = treated_units[treated_units > 0].index
    control_units  = treated_units.symmetric_difference(panel[unit_col].unique())

    # Compute treatment onset period per treated unit
    first_treat = panel[panel[treat_col] > 0].groupby(unit_col)[time_col].min()

    # Average outcomes by period × group
    mean_treat   = panel[panel[unit_col].isin(treated_units)].groupby(time_col)[outcome_col].mean()
    mean_control = panel[panel[unit_col].isin(control_units)].groupby(time_col)[outcome_col].mean()
    median_treat_start = int(first_treat.median()) if len(first_treat) else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean_treat.index,   mean_treat.values,   color="firebrick",  linewidth=2.0,
            label="Treated units (avg.)")
    ax.plot(mean_control.index, mean_control.values, color="steelblue", linewidth=2.0,
            linestyle="--", label="Control units (avg.)")

    if median_treat_start:
        ax.axvline(median_treat_start, color="black", linewidth=1.2,
                   linestyle=":", label=f"Treatment onset ({median_treat_start})")

    if not np.isnan(tau):
        ax.text(0.98, 0.05, f"τ̂_SDID = {tau:.4f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    ax.set_xlabel("Event period (year)")
    ax.set_ylabel(outcome_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=9)
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

    if not HAS_RPY2:
        print("WARNING: rpy2 not available. SDID requires rpy2 + synthdid R package.")
        print("  Install: rpy2 (pip), then in R: remotes::install_github('synth-inference/synthdid')")
        print("  Skipping SDID estimation. Writing placeholder table.")

    print("=== Analysis 08: SDID Estimates ===\n")

    df_a = pd.read_parquet(ANAL / "mechanism_a.parquet")
    df_b = pd.read_parquet(ANAL / "mechanism_b.parquet")

    results = []

    specs = [
        ("Mechanism A", df_a, "zcta5", "event_year",
         MECH_A_OUTCOME, "Contents damage ratio", MECH_A_TREAT,
         FIGS / "fig_sdid_mech_a.png"),
        ("Mechanism B", df_b, "zcta5", "event_year",
         MECH_B_OUTCOME, "IA application rate", MECH_B_TREAT,
         FIGS / "fig_sdid_mech_b.png"),
    ]

    for mech, df, unit_col, time_col, outcome_col, outcome_label, treat_col, fig_out in specs:
        print(f"\n--- {mech}: {outcome_label} ---")

        # Check required columns
        for col in [unit_col, time_col, outcome_col, treat_col]:
            if col not in df.columns:
                print(f"  SKIP: column '{col}' not found")
                results.append({"Mechanism": mech, "Outcome": outcome_label,
                                 "τ̂_SDID": "", "SE (placebo)": "", "Perm. p": "",
                                 "N units": "", "T periods": ""})
                break
        else:
            # Build balanced panel
            print("  Building balanced panel...")
            panel = build_balanced_panel(df, unit_col, time_col, outcome_col, treat_col)

            if panel is None or len(panel) == 0:
                print(f"  SKIP: could not build balanced panel")
                results.append({"Mechanism": mech, "Outcome": outcome_label,
                                 "τ̂_SDID": "—", "SE (placebo)": "—", "Perm. p": "—",
                                 "N units": "0", "T periods": "0"})
                continue

            n_units   = panel[unit_col].nunique()
            n_periods = panel[time_col].nunique()
            n_treated = panel.groupby(unit_col)[treat_col].max().gt(0).sum()

            if HAS_RPY2:
                print(f"  Running synthdid (N={n_units:,}, T={n_periods}, N_tr={n_treated})...")
                sdid_res = run_sdid_r(panel, unit_col, time_col, outcome_col, treat_col)

                if sdid_res.get("error"):
                    print(f"  ERROR: {sdid_res['error']}")
                    tau, se_p, perm_p = float("nan"), float("nan"), float("nan")
                else:
                    tau   = sdid_res["tau"]
                    se_p  = sdid_res["se_placebo"]
                    print(f"  τ̂_SDID = {tau:.4f}  SE(placebo) = {se_p:.4f}")

                    # Permutation inference (K=1000)
                    print(f"  Running permutation inference (K={N_PERMUTATIONS})...")
                    perm_p = run_sdid_permutation(
                        panel, unit_col, time_col, outcome_col, treat_col, tau
                    )
                    print(f"  Permutation p = {perm_p:.3f}" if not np.isnan(perm_p) else
                          "  Permutation p = N/A")

                # Path plot
                if not np.isnan(tau if 'tau' in dir() else float("nan")):
                    plot_sdid_path(
                        panel, unit_col, time_col, outcome_col, treat_col,
                        tau=tau,
                        title=f"SDID — {mech}: {outcome_label}",
                        outpath=fig_out,
                    )

                stars = ""
                if not np.isnan(sdid_res.get("p_value", float("nan")) if not sdid_res.get("error") else float("nan")):
                    pv = sdid_res["p_value"]
                    stars = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))

                results.append({
                    "Mechanism":   mech,
                    "Outcome":     outcome_label,
                    "τ̂_SDID":     f"{tau:.4f}{stars}" if not np.isnan(tau) else "—",
                    "SE (placebo)": f"{se_p:.4f}" if not np.isnan(se_p) else "—",
                    "Perm. p":     f"{perm_p:.3f}" if not np.isnan(perm_p) else "—",
                    "N units":     f"{n_units:,}",
                    "T periods":   str(n_periods),
                })
            else:
                results.append({
                    "Mechanism":   mech,
                    "Outcome":     outcome_label,
                    "τ̂_SDID":     "rpy2 unavailable",
                    "SE (placebo)": "", "Perm. p": "",
                    "N units":     f"{n_units:,}",
                    "T periods":   str(n_periods),
                })

    # --- LaTeX table ---
    tbl = pd.DataFrame(results)
    ncol = len(tbl.columns)
    col_fmt = "ll" + "r" * (ncol - 2)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Synthetic Difference-in-Differences Estimates}",
        r"\label{tab:sdid}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(tbl.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\footnotesize{{Note: SDID (Arkhangelsky et al. 2021) via \texttt{{synthdid}} R package. "
        rf"Panel restricted to ZIPs with $\geq${MIN_COVERAGE:.0%} period coverage; "
        rf"missing outcomes imputed to 0. "
        rf"SE(placebo) = leave-one-out placebo variance estimator. "
        rf"Perm. p = permutation p-value (K={N_PERMUTATIONS:,}). "
        rf"*** p$<$0.01, ** p$<$0.05, * p$<$0.10.}}",
        r"\end{table}",
    ]
    (TABLES / "tab_sdid.tex").write_text("\n".join(lines))
    print(f"\n  Saved: tables/tab_sdid.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
preprocessing/04b_construct_ttr_adjustment.py
TTR specification selection and threshold verification gate.

This script has two jobs:
  1. ENFORCE the data integrity rule: refuse to run if ia_thresholds_UNVERIFIED.csv
     contains any rows with verified=False.
  2. Compute all three TTR-adjusted running variable specifications (linear, log, rank)
     and run first-stage F-diagnostics to select the preferred specification.

Reads:
  data/raw/fema/ia_thresholds_UNVERIFIED.csv   ← HARD GATE: all rows must be verified=True
  data/raw/treasury/ttr_clean.csv               ← manually cleaned TTR data (must exist)
  data/intermediate/events_county_panel.parquet

Outputs:
  data/intermediate/ttr_spec_comparison.csv
  data/intermediate/events_county_panel_with_rv.parquet

TTR specifications (pre-specified in project plan §TTR-Adjusted Running Variable):
  Linear: (per_cap_damage / (ttr_pct_us_avg / 100)) - threshold
  Log:    per_cap_damage * ln(100 / ttr_pct_us_avg) - threshold
  Rank:   per_cap_damage * (1 + (1 - ttr_percentile_rank)) - threshold

Run order: after 02_construct_events.py; before 05_build_analysis_dataset.py
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"


def require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}\n  {hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Gate 0: Threshold verification
# ---------------------------------------------------------------------------

def check_threshold_verification() -> pd.DataFrame:
    """
    Hard gate: read ia_thresholds_UNVERIFIED.csv and refuse to proceed
    if any row has verified=False.
    """
    path = RAW / "fema" / "ia_thresholds_UNVERIFIED.csv"
    require(path, "Run acquire/05_ancillary.py")

    df = pd.read_csv(path)

    if "verified" not in df.columns:
        print("STOP: ia_thresholds_UNVERIFIED.csv has no 'verified' column.")
        print("  Recreate by running acquire/05_ancillary.py")
        sys.exit(1)

    unverified = df[df["verified"] != True]
    if not unverified.empty:
        print("=" * 60)
        print("STOP: IA threshold values have not been verified.")
        print()
        print(f"  {len(unverified)} rows have verified=False:")
        print(unverified[["year", "state_threshold", "county_threshold", "verified"]].to_string(index=False))
        print()
        print("ACTION REQUIRED:")
        print("  1. Look up each year's threshold in the Federal Register (44 CFR 206.48)")
        print("  2. Correct any values that differ from the published notice")
        print("  3. Set verified=True for each confirmed row")
        print("  4. Add Federal Register citation to 'source' column")
        print()
        print("  Federal Register: https://www.federalregister.gov/")
        print("  Search: 'FEMA Individual Assistance thresholds 44 CFR 206.48'")
        print("=" * 60)
        sys.exit(1)

    print(f"  Threshold verification: all {len(df)} rows verified ✓")
    return df[["year", "state_threshold", "county_threshold"]]


# ---------------------------------------------------------------------------
# Load and validate TTR data
# ---------------------------------------------------------------------------

def load_ttr() -> pd.DataFrame:
    path = RAW / "treasury" / "ttr_clean.csv"
    if not path.exists():
        print("STOP: data/raw/treasury/ttr_clean.csv not found.")
        print()
        print("  This file must be manually created from the Treasury TTR Excel download.")
        print("  1. Run acquire/05_ancillary.py to download ttr_raw.xlsx")
        print("  2. Inspect data/raw/treasury/ttr_raw_parsed.csv")
        print("  3. Produce ttr_clean.csv with columns:")
        print("       state_fips, fiscal_year, ttr_per_capita, ttr_pct_us_avg")
        print("  4. Verify values against Treasury TTR documentation")
        sys.exit(1)

    ttr = pd.read_csv(path)
    required_cols = {"ttr_pct_us_avg", "fiscal_year"}
    # Accept state_fips or state_name or state
    state_col = next((c for c in ttr.columns
                      if c.lower() in ("state_fips", "state", "state_name", "statename")), None)
    if not required_cols.issubset(set(ttr.columns)) or state_col is None:
        print(f"STOP: ttr_clean.csv missing required columns.")
        print(f"  Required: {required_cols | {'state_fips or state'}}")
        print(f"  Found: {set(ttr.columns)}")
        sys.exit(1)

    ttr = ttr.rename(columns={state_col: "state_fips"})
    ttr["ttr_pct_us_avg"] = pd.to_numeric(ttr["ttr_pct_us_avg"], errors="coerce")
    ttr["fiscal_year"]    = pd.to_numeric(ttr["fiscal_year"], errors="coerce")

    # Compute percentile rank within each fiscal year (cross-section of states)
    ttr["ttr_percentile_rank"] = (
        ttr.groupby("fiscal_year")["ttr_pct_us_avg"]
        .rank(pct=True, method="average")
    )

    print(f"  TTR data: {len(ttr):,} state-year observations, "
          f"years {int(ttr['fiscal_year'].min())}–{int(ttr['fiscal_year'].max())}")
    return ttr


# ---------------------------------------------------------------------------
# Compute TTR-adjusted running variables
# ---------------------------------------------------------------------------

def compute_running_variables(
    events: pd.DataFrame,
    thresholds: pd.DataFrame,
    ttr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute three TTR-adjusted running variable specifications per the project plan
    (fema-peltzman-project-plan-v2.md §TTR-Adjusted Running Variable).

    Uses county_threshold as the threshold value (county per-capita threshold).
    Threshold year = event_year; TTR fiscal_year matched to event_year.
    """
    df = events.copy()

    # Merge threshold by event year
    df = df.merge(
        thresholds.rename(columns={"year": "event_year"}),
        on="event_year", how="left",
    )

    # Merge TTR by state_fips × event_year (use event_year ≈ fiscal_year)
    # TTR is published by fiscal year (Oct–Sep); map event_year → fiscal_year
    df = df.merge(
        ttr[["state_fips", "fiscal_year", "ttr_pct_us_avg", "ttr_percentile_rank"]],
        left_on=["state_fips", "event_year"],
        right_on=["state_fips", "fiscal_year"],
        how="left",
    ).drop(columns=["fiscal_year"], errors="ignore")

    pc = df["per_capita_damage"]
    ttr_pct  = df["ttr_pct_us_avg"]
    ttr_rank = df["ttr_percentile_rank"]
    thresh   = df["county_threshold"]

    # Three TTR specifications (pre-specified in project plan §TTR-Adjusted Running Variable)
    # Linear: (per_cap_damage / (ttr_pct_us_avg / 100)) - threshold
    df["rv_linear"] = (pc / (ttr_pct / 100)) - thresh

    # Log: per_cap_damage * ln(100 / ttr_pct_us_avg) - threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rv_log"] = pc * np.log(100 / ttr_pct.replace(0, float("nan"))) - thresh

    # Rank: per_cap_damage * (1 + (1 - ttr_percentile_rank)) - threshold
    df["rv_rank"] = pc * (1 + (1 - ttr_rank)) - thresh

    n_missing = df[["rv_linear", "rv_log", "rv_rank"]].isna().all(axis=1).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rows missing all three RV specs "
              f"(likely missing TTR or threshold data for those years/states)")

    return df


# ---------------------------------------------------------------------------
# First-stage F-statistic diagnostics
# ---------------------------------------------------------------------------

def compute_first_stage_f(events_with_rv: pd.DataFrame) -> pd.DataFrame:
    """
    For each TTR specification, run OLS first stage:
      ia_declared ~ 1[rv_spec >= 0] + rv_spec + state_fips FE + year FE
    Report F-statistic and N.
    """
    results = []

    for spec in ["rv_linear", "rv_log", "rv_rank"]:
        if spec not in events_with_rv.columns:
            continue
        df = events_with_rv[
            [spec, "ia_declared", "state_fips", "event_year"]
        ].dropna().copy()

        if len(df) < 50:
            results.append({"spec": spec, "n": len(df), "f_stat": float("nan"),
                             "note": "too few observations"})
            continue

        df["threshold_crossing"] = (df[spec] >= 0).astype(int)

        # OLS with state and year FE via dummy variables (approximate for diagnostics)
        # For the full analysis, linearmodels.IV2SLS is used
        try:
            formula = f"ia_declared ~ threshold_crossing + {spec} + C(state_fips) + C(event_year)"
            mod = smf.ols(formula, data=df).fit()

            # F-stat for threshold_crossing coefficient
            coef = mod.params.get("threshold_crossing", float("nan"))
            se   = mod.bse.get("threshold_crossing", float("nan"))
            f    = (coef / se) ** 2 if se > 0 else float("nan")

            results.append({
                "spec":         spec,
                "n":            int(len(df)),
                "f_stat":       round(float(f), 2) if not np.isnan(f) else float("nan"),
                "coef":         round(float(coef), 4),
                "se":           round(float(se), 4),
                "r_squared":    round(float(mod.rsquared), 4),
                "note":         "OK",
            })
        except Exception as e:
            results.append({"spec": spec, "n": len(df), "f_stat": float("nan"),
                             "note": str(e)[:80]})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_rv   = INTER / "events_county_panel_with_rv.parquet"
    out_comp = INTER / "ttr_spec_comparison.csv"

    if out_rv.exists() and out_comp.exists():
        print("events_county_panel_with_rv.parquet already exists — delete to rebuild")
        comp = pd.read_csv(out_comp)
        print("\nTTR spec comparison:")
        print(comp.to_string(index=False))
        return

    print("=== Step 04b: TTR Adjustment ===\n")

    require(INTER / "events_county_panel.parquet", "Run 02_construct_events.py")

    # GATE: threshold verification
    print("Checking threshold verification...")
    thresholds = check_threshold_verification()

    # Load TTR
    print("\nLoading Treasury TTR data...")
    ttr = load_ttr()

    # Load events
    events = pd.read_parquet(INTER / "events_county_panel.parquet")
    print(f"Events panel: {len(events):,} rows")

    # Compute running variables
    print("\nComputing TTR-adjusted running variables (3 specifications)...")
    events_rv = compute_running_variables(events, thresholds, ttr)

    # First-stage F diagnostics
    print("\nRunning first-stage diagnostics...")
    comparison = compute_first_stage_f(events_rv)

    print("\n" + "=" * 55)
    print("TTR Specification Comparison — First-Stage F")
    print("=" * 55)
    print(f"{'Spec':<12} {'N':>8} {'F-stat':>8} {'Coef':>8} {'SE':>8}  Note")
    print("-" * 55)
    for _, row in comparison.iterrows():
        f = f"{row['f_stat']:.2f}" if pd.notna(row.get("f_stat")) else "  N/A"
        c = f"{row.get('coef', float('nan')):.4f}" if pd.notna(row.get("coef")) else "  N/A"
        s = f"{row.get('se', float('nan')):.4f}" if pd.notna(row.get("se")) else "  N/A"
        print(f"{row['spec']:<12} {int(row['n']):>8} {f:>8} {c:>8} {s:>8}  {row.get('note','')}")
    print("=" * 55)

    if comparison["f_stat"].notna().any():
        best_spec = comparison.loc[comparison["f_stat"].idxmax(), "spec"]
        best_f    = comparison["f_stat"].max()
        print(f"\nPreferred specification: {best_spec} (F={best_f:.2f})")
        if best_f < 10:
            print("WARNING: Max F-stat < 10. This is a gate-check criterion.")
            print("  analysis/00_gate_check.py will flag this as a potential stop.")
        events_rv["rv_preferred"] = events_rv[best_spec]
        events_rv["rv_preferred_spec"] = best_spec
    else:
        print("\nWARNING: Could not compute F-statistics (possibly no TTR data coverage)")
        events_rv["rv_preferred"] = events_rv.get("rv_linear")
        events_rv["rv_preferred_spec"] = "rv_linear_fallback"

    # Save outputs
    comparison.to_csv(out_comp, index=False)
    events_rv.to_parquet(out_rv, index=False)
    print(f"\nSaved ttr_spec_comparison.csv → {out_comp.relative_to(ROOT)}")
    print(f"Saved events_county_panel_with_rv.parquet → {out_rv.relative_to(ROOT)}")

    # Running variable distribution check
    for spec in ["rv_linear", "rv_log", "rv_rank"]:
        if spec in events_rv.columns:
            v = events_rv[spec].dropna()
            print(f"\n  {spec}: mean={v.mean():.3f}, median={v.median():.3f}, "
                  f"pct_positive={v.gt(0).mean():.1%}")
            if abs(v.mean()) > 5:
                print(f"  WARNING: {spec} mean is far from 0 — check TTR/threshold alignment")


if __name__ == "__main__":
    main()

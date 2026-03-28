"""
preprocessing/05_build_analysis_dataset.py
Merge all intermediate files and apply sample restrictions.
Produces the four analysis-ready parquets consumed by all analysis scripts.

Reads:
  data/intermediate/events_county_panel_with_rv.parquet
  data/intermediate/zip_outcomes_panel.parquet
  data/intermediate/hazard_intensity_panel.parquet
  data/intermediate/zcta_county_primary.parquet
  data/intermediate/ttr_spec_comparison.csv

Outputs (data/analysis/):
  mechanism_a.parquet        — Mechanism A, all hazards
  mechanism_a_flood.parquet  — Mechanism A, flood-only
  mechanism_b.parquet        — Mechanism B, all hazards
  mechanism_b_flood.parquet  — Mechanism B, flood-only

Sample restrictions per project plan §Step 5:
  Dataset A: prior IA linkage exists, years_since_prior ≤ 10, adequate_notice,
             contents_damage_ratio not null
  Dataset B: within bandwidth of contemporaneous RV, has NFIP or IA activity

Run order: after 02, 03, 04, 04b.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"
ANAL  = ROOT / "data" / "analysis"
ANAL.mkdir(exist_ok=True)

# Maximum years since prior event for Mechanism A primary sample
# (pre-specified in project plan §Step 5, citing Gallagher 2014)
MECH_A_MAX_YEARS_SINCE_PRIOR = 10

# Initial bandwidth for sample restriction (very wide — CCT optimal computed in analysis)
# This is a data filter, not an analytical bandwidth. Pre-specified in project plan.
INITIAL_BANDWIDTH_PER_CAPITA = 5.0  # $5/capita around threshold


def require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}\n  {hint}")
        sys.exit(1)


def load_preferred_spec() -> str:
    comp_path = INTER / "ttr_spec_comparison.csv"
    if not comp_path.exists():
        print("  WARNING: ttr_spec_comparison.csv not found — defaulting to rv_linear")
        return "rv_linear"
    comp = pd.read_csv(comp_path)
    if comp.empty or "f_stat" not in comp.columns:
        return "rv_linear"
    valid = comp[comp["f_stat"].notna()]
    if valid.empty:
        return "rv_linear"
    return valid.loc[valid["f_stat"].idxmax(), "spec"]


# ---------------------------------------------------------------------------
# Merge all intermediates
# ---------------------------------------------------------------------------

def build_merged_panel() -> pd.DataFrame:
    require(INTER / "events_county_panel_with_rv.parquet", "Run 04b_construct_ttr_adjustment.py")
    require(INTER / "zip_outcomes_panel.parquet",          "Run 03_construct_zip_outcomes.py")
    require(INTER / "hazard_intensity_panel.parquet",      "Run 04_hazard_intensity.py")
    require(INTER / "zcta_county_primary.parquet",         "Run 01_standardize_geography.py")

    events   = pd.read_parquet(INTER / "events_county_panel_with_rv.parquet")
    outcomes = pd.read_parquet(INTER / "zip_outcomes_panel.parquet")
    hazard   = pd.read_parquet(INTER / "hazard_intensity_panel.parquet")
    xwalk    = pd.read_parquet(INTER / "zcta_county_primary.parquet")

    print(f"  events_county_panel_with_rv:  {len(events):,} rows")
    print(f"  zip_outcomes_panel:           {len(outcomes):,} rows")
    print(f"  hazard_intensity_panel:       {len(hazard):,} rows")

    # outcomes is at ZIP × disaster level; events is at county × disaster
    # Link: ZIP → county via xwalk, then merge events
    if "county_fips" not in outcomes.columns:
        outcomes = outcomes.merge(xwalk[["zcta5", "county_fips"]], on="zcta5", how="left")

    # Merge events to outcomes via county × disasterNumber
    merged = outcomes.merge(
        events.drop(columns=["zcta5"], errors="ignore"),
        on=["disasterNumber", "county_fips"],
        how="left",
    )

    # Merge hazard intensity (county × disaster level)
    merged = merged.merge(
        hazard, on=["disasterNumber", "county_fips"], how="left",
    )

    print(f"  Merged panel:                 {len(merged):,} rows")
    return merged


# ---------------------------------------------------------------------------
# Build Dataset A — Mechanism A (Historical IV, Pre-Disaster Mitigation)
# ---------------------------------------------------------------------------

def build_dataset_a(merged: pd.DataFrame, preferred_rv: str) -> pd.DataFrame:
    """
    Restrictions (pre-specified in project plan §Step 5):
    1. prior_ia_received is not null (has prior event linkage)
    2. years_since_prior <= MECH_A_MAX_YEARS_SINCE_PRIOR (10 years per Gallagher)
    3. adequate_notice == True (warning_lead_hours >= 24)
    4. contents_damage_ratio is not null (primary outcome must exist)
    5. county_population > 0 and ACS demographics not entirely null
    6. Prior running variable within initial bandwidth of threshold
    """
    df = merged.copy()
    n0 = len(df)

    # 1. Has prior event linkage
    mask = df["prior_ia_received"].notna()
    df   = df[mask]
    print(f"  Has prior linkage:      {len(df):,} / {n0:,}")

    # 2. Years since prior ≤ 10
    if "years_since_prior" in df.columns:
        mask = df["years_since_prior"].le(MECH_A_MAX_YEARS_SINCE_PRIOR) | df["years_since_prior"].isna()
        df   = df[df["years_since_prior"].notna() & df["years_since_prior"].le(MECH_A_MAX_YEARS_SINCE_PRIOR)]
        print(f"  years_since_prior ≤ {MECH_A_MAX_YEARS_SINCE_PRIOR}:   {len(df):,}")

    # 3. Adequate warning notice (≥24h)
    if "adequate_notice" in df.columns:
        df = df[df["adequate_notice"] == True]
        print(f"  adequate_notice (≥24h): {len(df):,}")

    # 4. Contents damage ratio not null (primary Mechanism A outcome)
    if "contents_damage_ratio" in df.columns:
        df = df[df["contents_damage_ratio"].notna()]
        print(f"  contents_ratio present: {len(df):,}")

    # 5. Positive population
    if "county_population" in df.columns:
        df = df[df["county_population"].fillna(0).gt(0)]
        print(f"  population > 0:         {len(df):,}")

    # 6. Prior running variable within initial bandwidth
    prior_rv_col = f"prior_running_var_raw"
    if prior_rv_col in df.columns:
        df = df[df[prior_rv_col].abs().le(INITIAL_BANDWIDTH_PER_CAPITA)]
        print(f"  prior RV within ±{INITIAL_BANDWIDTH_PER_CAPITA}:    {len(df):,}")

    # Rename preferred RV columns for clarity
    if "rv_preferred" in df.columns:
        df = df.rename(columns={"rv_preferred": "running_var_preferred"})
    if preferred_rv in df.columns:
        df["running_var"] = df[preferred_rv]

    n_final = len(df)
    if n_final < 500:
        print(f"\n  *** WARNING: Mechanism A has only {n_final} observations. ***")
        print(f"  Gate check (analysis/00_gate_check.py) requires N ≥ 500.")
        print(f"  Consider relaxing bandwidth or warning lead time threshold.")

    return df


# ---------------------------------------------------------------------------
# Build Dataset B — Mechanism B (Contemporaneous RD)
# ---------------------------------------------------------------------------

def build_dataset_b(merged: pd.DataFrame, preferred_rv: str) -> pd.DataFrame:
    """
    Restrictions (pre-specified in project plan §Step 5):
    1. Current event within initial bandwidth of contemporaneous RV
    2. Has some claim activity (NFIP or IA)
    3. Positive population
    """
    df = merged.copy()
    n0 = len(df)

    # 1. Within initial bandwidth of contemporaneous running variable
    rv_col = preferred_rv if preferred_rv in df.columns else "running_var_raw"
    if rv_col in df.columns:
        df = df[df[rv_col].notna() & df[rv_col].abs().le(INITIAL_BANDWIDTH_PER_CAPITA)]
        print(f"  Within ±{INITIAL_BANDWIDTH_PER_CAPITA} of threshold: {len(df):,} / {n0:,}")

    # 2. Has NFIP or IA activity
    has_nfip = df.get("n_nfip_claims", pd.Series(0)).fillna(0).gt(0)
    has_ia   = df.get("n_total_registrations", pd.Series(0)).fillna(0).gt(0)
    df = df[has_nfip | has_ia]
    print(f"  Has NFIP or IA activity: {len(df):,}")

    # 3. Positive population
    if "county_population" in df.columns:
        df = df[df["county_population"].fillna(0).gt(0)]
        print(f"  population > 0:          {len(df):,}")

    # Add treatment indicator and running variable alias
    if "ia_declared" in df.columns:
        df["treatment"] = df["ia_declared"]
    if rv_col in df.columns:
        df["running_var"] = df[rv_col]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_a       = ANAL / "mechanism_a.parquet"
    out_a_flood = ANAL / "mechanism_a_flood.parquet"
    out_b       = ANAL / "mechanism_b.parquet"
    out_b_flood = ANAL / "mechanism_b_flood.parquet"

    if all(p.exists() for p in [out_a, out_a_flood, out_b, out_b_flood]):
        print("All four analysis parquets already exist — delete to rebuild")
        for p in [out_a, out_a_flood, out_b, out_b_flood]:
            df = pd.read_parquet(p)
            print(f"  {p.name}: {len(df):,} rows")
        return

    print("=== Step 5: Build Analysis Datasets ===\n")

    preferred_rv = load_preferred_spec()
    print(f"Preferred TTR specification: {preferred_rv}\n")

    # Merge all intermediates
    print("Merging all intermediate files...")
    merged = build_merged_panel()

    # Dataset A
    print("\n--- Dataset A: Mechanism A (all hazards) ---")
    ds_a = build_dataset_a(merged, preferred_rv)
    ds_a.to_parquet(out_a, index=False)
    print(f"  → Saved {len(ds_a):,} rows to mechanism_a.parquet")

    # Dataset A_Flood
    print("\n--- Dataset A_Flood: Mechanism A (flood-only) ---")
    if "hazard_type" in ds_a.columns:
        ds_a_flood = ds_a[ds_a["hazard_type"].isin(["flood_only", "flood_mixed"])]
    elif "incidentType" in ds_a.columns:
        ds_a_flood = ds_a[ds_a["incidentType"].str.lower().isin(["flood", "coastal storm"])]
    else:
        ds_a_flood = ds_a.iloc[0:0]  # empty if no hazard type info
    ds_a_flood.to_parquet(out_a_flood, index=False)
    print(f"  → Saved {len(ds_a_flood):,} rows to mechanism_a_flood.parquet")

    # Dataset B
    print("\n--- Dataset B: Mechanism B (all hazards) ---")
    ds_b = build_dataset_b(merged, preferred_rv)
    ds_b.to_parquet(out_b, index=False)
    print(f"  → Saved {len(ds_b):,} rows to mechanism_b.parquet")

    # Dataset B_Flood
    print("\n--- Dataset B_Flood: Mechanism B (flood-only) ---")
    if "hazard_type" in ds_b.columns:
        ds_b_flood = ds_b[ds_b["hazard_type"].isin(["flood_only", "flood_mixed"])]
    elif "incidentType" in ds_b.columns:
        ds_b_flood = ds_b[ds_b["incidentType"].str.lower().isin(["flood", "coastal storm"])]
    else:
        ds_b_flood = ds_b.iloc[0:0]
    ds_b_flood.to_parquet(out_b_flood, index=False)
    print(f"  → Saved {len(ds_b_flood):,} rows to mechanism_b_flood.parquet")

    # Final summary
    print("\n" + "=" * 50)
    print("ANALYSIS DATASET SUMMARY")
    print("=" * 50)
    datasets = {
        "mechanism_a":       ds_a,
        "mechanism_a_flood": ds_a_flood,
        "mechanism_b":       ds_b,
        "mechanism_b_flood": ds_b_flood,
    }
    for name, ds in datasets.items():
        n = len(ds)
        ia_share = ds["ia_declared"].mean() if "ia_declared" in ds.columns else float("nan")
        flag = " ← WARNING: N < 500" if n < 500 else ""
        print(f"  {name:<22} {n:>7,} rows  IA={ia_share:.1%}{flag}")

    print("\nNext: run analysis/00_gate_check.py")
    if len(ds_a) < 500:
        print("\n*** GATE ALERT: mechanism_a has <500 rows. ***")
        print("    analysis/00_gate_check.py will likely fail Gate 4.")
        print("    Consider relaxing sample restrictions in this script.")


if __name__ == "__main__":
    main()

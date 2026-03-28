"""
preprocessing/03_construct_zip_outcomes.py
Aggregate NFIP and IA claims to ZIP × event level; compute all outcome variables.

Reads:
  data/raw/fema/nfip_claims.parquet
  data/raw/fema/housing_assistance_owners.parquet
  data/raw/fema/housing_assistance_renters.parquet
  data/raw/fema/ia_registrants_large_disasters.parquet
  data/raw/census/acs_zcta_demographics.parquet
  data/intermediate/events_county_panel.parquet
  data/intermediate/zcta_county_primary.parquet

Outputs:
  data/intermediate/zip_outcomes_panel.parquet

Run order: after 01_standardize_geography.py and 02_construct_events.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"

ACS_VARS_NEEDED = [
    "zip code tabulation area",
    "acs_year",
    "B01003_001E",  # population
    "B25001_001E",  # total housing units
    "B25003_001E",  # tenure total
    "B25003_002E",  # owner-occupied
    "B25003_003E",  # renter-occupied
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25002_002E",  # occupied housing units
    "B02001_002E",  # white alone
    "B02001_001E",  # race total
    "B03001_003E",  # Hispanic
    "B03001_001E",  # Hispanic origin total
    "B01002_001E",  # median age
]


def require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}\n  {hint}")
        sys.exit(1)


def get_acs_vintage(event_year: int, acs_years: list) -> int:
    candidates = [y for y in acs_years if y <= event_year]
    return max(candidates) if candidates else min(acs_years)


# ---------------------------------------------------------------------------
# 3a. Match NFIP claims to events
# ---------------------------------------------------------------------------

def load_nfip_claims_by_event(events: pd.DataFrame) -> pd.DataFrame:
    path = RAW / "fema" / "nfip_claims.parquet"
    require(path, "Run acquire/01_fema.py")

    print("  Loading NFIP claims...")
    # Read only needed columns
    nfip_cols = [
        "reportedZipCode", "dateOfLoss", "disasterNumber",
        "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
        "buildingDamageAmount", "contentsDamageAmount",
        "waterDepth", "floodZone",
    ]
    available = pd.read_parquet(path, columns=["reportedZipCode"]).columns  # probe
    actual_cols = [c for c in nfip_cols if c in pd.read_parquet(path).columns]

    nfip = pd.read_parquet(path, columns=actual_cols)
    print(f"  NFIP claims: {len(nfip):,} rows")

    nfip["zcta5"] = nfip["reportedZipCode"].astype(str).str.strip().str.zfill(5)
    nfip["dateOfLoss"] = pd.to_datetime(nfip["dateOfLoss"], errors="coerce", utc=True).dt.tz_localize(None)
    nfip["loss_year"] = nfip["dateOfLoss"].dt.year

    for col in ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
                "buildingDamageAmount", "contentsDamageAmount", "waterDepth"]:
        if col in nfip.columns:
            nfip[col] = pd.to_numeric(nfip[col], errors="coerce")

    # Match to events: prefer disasterNumber if available, else date+county proximity
    if "disasterNumber" in nfip.columns:
        nfip["disasterNumber"] = pd.to_numeric(nfip["disasterNumber"], errors="coerce")
        # Merge on disasterNumber
        event_keys = events[["disasterNumber", "county_fips", "incidentBeginDate",
                              "incidentEndDate", "ia_declared", "event_year"]].drop_duplicates()
        xwalk = pd.read_parquet(INTER / "zcta_county_primary.parquet")[["zcta5", "county_fips"]]
        nfip_with_county = nfip.merge(xwalk, on="zcta5", how="left")
        matched = nfip_with_county.merge(
            event_keys, on=["disasterNumber", "county_fips"], how="inner"
        )
        print(f"  NFIP claims matched by disasterNumber: {len(matched):,}")
    else:
        # Date-based matching: assign claim to event with overlapping date range in same county
        print("  No disasterNumber in NFIP — using date-based matching")
        xwalk = pd.read_parquet(INTER / "zcta_county_primary.parquet")[["zcta5", "county_fips"]]
        nfip_with_county = nfip.merge(xwalk, on="zcta5", how="left")
        event_keys = events[["disasterNumber", "county_fips", "incidentBeginDate",
                              "incidentEndDate", "ia_declared", "event_year"]].copy()
        event_keys["event_window_start"] = event_keys["incidentBeginDate"] - pd.Timedelta(days=7)
        event_keys["event_window_end"]   = event_keys["incidentEndDate"].fillna(
            event_keys["incidentBeginDate"]
        ) + pd.Timedelta(days=90)  # NFIP claims often filed weeks after event

        # Merge on county then filter by date window
        joined = nfip_with_county.merge(event_keys, on="county_fips", how="inner")
        matched = joined[
            (joined["dateOfLoss"] >= joined["event_window_start"]) &
            (joined["dateOfLoss"] <= joined["event_window_end"])
        ].copy()
        print(f"  NFIP claims matched by date: {len(matched):,}")

    return matched


# ---------------------------------------------------------------------------
# 3b. Mechanism A outcomes per ZIP × event
# ---------------------------------------------------------------------------

def compute_mechanism_a_outcomes(nfip_matched: pd.DataFrame) -> pd.DataFrame:
    """
    contents_damage_ratio: fraction of total paid that is contents (mitigation proxy).
    building_paid_per_depth_ft: residual structural damage per foot of water depth.
    total_claim_severity: total paid (building + contents); normalized by hazard in 04.
    """
    building_col  = "amountPaidOnBuildingClaim"
    contents_col  = "amountPaidOnContentsClaim"
    depth_col     = "waterDepth"

    df = nfip_matched.copy()

    # Exclude rows where both paid amounts are zero or missing
    has_paid = (
        df[building_col].fillna(0).gt(0) | df[contents_col].fillna(0).gt(0)
    ) if building_col in df.columns else pd.Series(True, index=df.index)

    df = df[has_paid].copy()
    df["total_paid"] = df.get(building_col, 0).fillna(0) + df.get(contents_col, 0).fillna(0)
    df["contents_damage_ratio"] = (
        df.get(contents_col, pd.Series(0, index=df.index)).fillna(0) /
        df["total_paid"].replace(0, float("nan"))
    )

    # Depth-adjusted building damage
    if depth_col in df.columns:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
        df["building_paid_per_depth_ft"] = (
            df.get(building_col, pd.Series(0, index=df.index)).fillna(0) /
            df[depth_col].replace(0, float("nan"))
        )
    else:
        df["building_paid_per_depth_ft"] = float("nan")

    # Aggregate to ZIP × disaster
    group_cols = [c for c in ["zcta5", "disasterNumber"] if c in df.columns]
    agg = df.groupby(group_cols).agg(
        n_nfip_claims          =("total_paid", "count"),
        contents_damage_ratio  =("contents_damage_ratio", "mean"),
        building_paid_per_depth_ft=("building_paid_per_depth_ft", "mean"),
        total_claim_severity   =("total_paid", "sum"),
        mean_water_depth_ft    =(depth_col, "mean") if depth_col in df.columns
                                 else ("total_paid", lambda x: float("nan")),
    ).reset_index()

    # Validation
    ratio = agg["contents_damage_ratio"]
    out_of_range = ratio[ratio.notna() & (~ratio.between(0, 1))].shape[0]
    if out_of_range > 0:
        print(f"  WARNING: {out_of_range} rows with contents_damage_ratio outside [0,1]")

    return agg


# ---------------------------------------------------------------------------
# 3c. Mechanism B outcomes per ZIP × event
# ---------------------------------------------------------------------------

def compute_mechanism_b_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    """
    ia_application_rate, ia_approval_rate, avg_ia_award (owner + renter),
    time_to_ia_application if timestamps available.
    """
    # Housing Assistance (ZIP-level native)
    ha_owners_path  = RAW / "fema" / "housing_assistance_owners.parquet"
    ha_renters_path = RAW / "fema" / "housing_assistance_renters.parquet"

    dfs_ha = []
    for path, tenure in [(ha_owners_path, "owner"), (ha_renters_path, "renter")]:
        if not path.exists():
            print(f"  WARNING: {path.name} not found — {tenure} outcomes unavailable")
            continue
        ha = pd.read_parquet(path)
        ha["tenure"] = tenure

        # Detect key columns
        zip_col      = next((c for c in ha.columns if "zip" in c.lower()), None)
        disaster_col = "disasterNumber"
        reg_col      = next((c for c in ha.columns
                             if "valid" in c.lower() and "reg" in c.lower()), None)
        approved_col = next((c for c in ha.columns
                             if "approved" in c.lower() and "assist" in c.lower()), None)
        amount_col   = next((c for c in ha.columns
                             if "total" in c.lower() and ("approved" in c.lower() or
                             "ihp" in c.lower() or "amount" in c.lower())), None)

        if not zip_col:
            print(f"  WARNING: no ZIP column in {path.name} — skipping")
            continue

        ha["zcta5"] = ha[zip_col].astype(str).str.strip().str.zfill(5)
        for col in [reg_col, approved_col, amount_col]:
            if col:
                ha[col] = pd.to_numeric(ha[col], errors="coerce")

        dfs_ha.append(ha.rename(columns={
            reg_col:      f"n_registrations_{tenure}",
            approved_col: f"n_approved_{tenure}",
            amount_col:   f"total_approved_amount_{tenure}",
        }))

    if not dfs_ha:
        print("  WARNING: No Housing Assistance data — Mechanism B outcomes unavailable")
        return pd.DataFrame(columns=["zcta5", "disasterNumber"])

    # Merge owner + renter
    if len(dfs_ha) == 2:
        key_cols = ["zcta5", "disasterNumber"]
        b_outcomes = dfs_ha[0][key_cols + [c for c in dfs_ha[0].columns
                                           if c not in key_cols]].merge(
            dfs_ha[1][key_cols + [c for c in dfs_ha[1].columns
                                  if c not in key_cols]],
            on=key_cols, how="outer",
        )
    else:
        b_outcomes = dfs_ha[0]

    # Aggregate rates
    b_outcomes["n_total_registrations"] = (
        b_outcomes.get("n_registrations_owner", pd.Series(0)).fillna(0) +
        b_outcomes.get("n_registrations_renter", pd.Series(0)).fillna(0)
    )
    b_outcomes["n_total_approved"] = (
        b_outcomes.get("n_approved_owner", pd.Series(0)).fillna(0) +
        b_outcomes.get("n_approved_renter", pd.Series(0)).fillna(0)
    )
    b_outcomes["total_approved_amount"] = (
        b_outcomes.get("total_approved_amount_owner", pd.Series(0)).fillna(0) +
        b_outcomes.get("total_approved_amount_renter", pd.Series(0)).fillna(0)
    )
    b_outcomes["ia_approval_rate"] = (
        b_outcomes["n_total_approved"] /
        b_outcomes["n_total_registrations"].replace(0, float("nan"))
    )
    b_outcomes["avg_ia_award"] = (
        b_outcomes["total_approved_amount"] /
        b_outcomes["n_total_approved"].replace(0, float("nan"))
    )

    print(f"  Housing Assistance outcomes: {len(b_outcomes):,} ZIP-events")
    return b_outcomes


# ---------------------------------------------------------------------------
# 3d. ACS demographics at ZCTA level
# ---------------------------------------------------------------------------

def load_acs_demographics() -> pd.DataFrame:
    path = RAW / "census" / "acs_zcta_demographics.parquet"
    require(path, "Run acquire/04_census.py")

    available_cols = pd.read_parquet(path).columns.tolist()
    cols_to_load = [c for c in ACS_VARS_NEEDED if c in available_cols]
    acs = pd.read_parquet(path, columns=cols_to_load)

    acs = acs.rename(columns={"zip code tabulation area": "zcta5"})
    acs["zcta5"] = acs["zcta5"].astype(str).str.zfill(5)

    for col in [c for c in acs.columns if c not in ("zcta5", "acs_year")]:
        acs[col] = pd.to_numeric(acs[col], errors="coerce")

    # Compute derived variables
    if "B25003_002E" in acs.columns and "B25003_001E" in acs.columns:
        acs["pct_owner_occ"] = acs["B25003_002E"] / acs["B25003_001E"].replace(0, float("nan"))
    if "B02001_002E" in acs.columns and "B02001_001E" in acs.columns:
        acs["pct_white"] = acs["B02001_002E"] / acs["B02001_001E"].replace(0, float("nan"))
    if "B03001_003E" in acs.columns and "B03001_001E" in acs.columns:
        acs["pct_hispanic"] = acs["B03001_003E"] / acs["B03001_001E"].replace(0, float("nan"))

    # Rename for clarity
    acs = acs.rename(columns={
        "B01003_001E": "population",
        "B25001_001E": "housing_units",
        "B19013_001E": "median_hh_income",
        "B25077_001E": "median_home_value",
        "B01002_001E": "median_age",
    })
    return acs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = INTER / "zip_outcomes_panel.parquet"
    if out.exists():
        print("zip_outcomes_panel.parquet already exists — delete to rebuild")
        df = pd.read_parquet(out)
        print(f"  {len(df):,} rows, columns: {list(df.columns)}")
        return

    print("=== Step 3: Construct ZIP Outcomes ===\n")

    require(INTER / "events_county_panel.parquet",  "Run 02_construct_events.py")
    require(INTER / "zcta_county_primary.parquet",  "Run 01_standardize_geography.py")

    events = pd.read_parquet(INTER / "events_county_panel.parquet")
    print(f"Events panel: {len(events):,} rows")

    # 3a. NFIP claims → Mechanism A outcomes
    print("\n3a. Mechanism A outcomes (NFIP claims)...")
    nfip_matched = load_nfip_claims_by_event(events)
    mech_a = compute_mechanism_a_outcomes(nfip_matched)
    print(f"  Mechanism A outcomes: {len(mech_a):,} ZIP-events")

    # 3b. Mechanism B outcomes (IA applications)
    print("\n3b. Mechanism B outcomes (IA applications)...")
    mech_b = compute_mechanism_b_outcomes(events)

    # 3c. ACS demographics
    print("\n3c. Loading ACS demographics...")
    acs = load_acs_demographics()
    acs_years = sorted(acs["acs_year"].unique())
    print(f"  ACS: {len(acs):,} ZCTA-year observations, years: {acs_years[0]}–{acs_years[-1]}")

    # Merge housing units from ACS into Mechanism B for application rate
    # Get all ZCTAs × event years needing ACS vintage
    all_zip_events = pd.concat([
        mech_a[["zcta5", "disasterNumber"]],
        mech_b[["zcta5", "disasterNumber"]] if "zcta5" in mech_b.columns else pd.DataFrame(),
    ]).drop_duplicates()

    all_zip_events = all_zip_events.merge(
        events[["disasterNumber", "event_year"]].drop_duplicates(),
        on="disasterNumber", how="left",
    )
    all_zip_events["acs_vintage"] = all_zip_events["event_year"].apply(
        lambda y: get_acs_vintage(int(y), acs_years) if pd.notna(y) else None
    )
    all_zip_events = all_zip_events.merge(
        acs, left_on=["zcta5", "acs_vintage"], right_on=["zcta5", "acs_year"], how="left",
    ).drop(columns=["acs_year"], errors="ignore")

    # Compute ia_application_rate = registrations / housing units
    if "n_total_registrations" in mech_b.columns and "zcta5" in mech_b.columns:
        hu_lookup = all_zip_events[["zcta5", "disasterNumber", "housing_units"]].drop_duplicates()
        mech_b = mech_b.merge(hu_lookup, on=["zcta5", "disasterNumber"], how="left")
        mech_b["ia_application_rate"] = (
            mech_b["n_total_registrations"] /
            mech_b["housing_units"].replace(0, float("nan"))
        )

    # Merge everything: start from all_zip_events (has ACS), add A and B outcomes
    result = all_zip_events.merge(mech_a, on=["zcta5", "disasterNumber"], how="left")

    b_merge_cols = [c for c in mech_b.columns
                    if c not in result.columns or c in ("zcta5", "disasterNumber")]
    if "zcta5" in mech_b.columns and "disasterNumber" in mech_b.columns:
        result = result.merge(
            mech_b[list(set(["zcta5", "disasterNumber"] + b_merge_cols))],
            on=["zcta5", "disasterNumber"], how="left",
        )

    result.to_parquet(out, index=False)
    print(f"\nSaved {len(result):,} rows → {out.relative_to(ROOT)}")

    # Validation
    print("\n--- Validation ---")
    if "contents_damage_ratio" in result.columns:
        r = result["contents_damage_ratio"]
        print(f"  contents_damage_ratio: "
              f"mean={r.mean():.3f}, "
              f"missing={r.isna().mean():.1%}, "
              f"out_of_range={r[r.notna() & (~r.between(0, 1))].shape[0]}")
    if "ia_application_rate" in result.columns:
        r = result["ia_application_rate"]
        print(f"  ia_application_rate:   "
              f"mean={r.mean():.3f}, "
              f"nonzero={r.gt(0).sum():,}")
    if "pct_owner_occ" in result.columns:
        r = result["pct_owner_occ"]
        print(f"  pct_owner_occ:         mean={r.mean():.3f}, missing={r.isna().mean():.1%}")


def get_acs_vintage(event_year, acs_years):
    candidates = [y for y in acs_years if y <= event_year]
    return max(candidates) if candidates else min(acs_years)


if __name__ == "__main__":
    main()

"""
preprocessing/01_standardize_geography.py
Build the geographic backbone used by all downstream scripts.

Reads:
  data/raw/census/zcta_county_crosswalk.csv
  data/raw/census/zcta_shapefiles/tl_2020_us_zcta520.zip
  data/raw/fema/disaster_declarations.parquet
  data/raw/noaa/storm_events_details.parquet

Outputs (data/intermediate/):
  zcta_county_primary.parquet   — each ZCTA → primary county (largest land share)
  zcta_centroids.parquet        — ZCTA centroid lat/lon (WGS84)
  fips_standardized.parquet     — FIPS reconciliation table across FEMA/NOAA/Census

Run order: first preprocessing script; no intermediate dependencies.
"""

import sys
import zipfile
import geopandas as gpd
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

RAW       = ROOT / "data" / "raw"
INTER     = ROOT / "data" / "intermediate"
INTER.mkdir(exist_ok=True)


def require(path: Path, label: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}")
        print(f"  Run acquire/04_census.py to download {label}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. ZCTA → primary county
# ---------------------------------------------------------------------------

def build_zcta_county_primary() -> pd.DataFrame:
    out = INTER / "zcta_county_primary.parquet"
    if out.exists():
        print("  zcta_county_primary.parquet already exists — loading")
        return pd.read_parquet(out)

    xwalk_path = RAW / "census" / "zcta_county_crosswalk.csv"
    require(xwalk_path, "ZCTA-to-county crosswalk")

    print("  Reading ZCTA-to-county relationship file...")
    xwalk = pd.read_csv(xwalk_path, dtype=str, low_memory=False)

    # Census 2020 relationship file columns:
    # GEOID_ZCTA5_20, GEOID_COUNTY_20, AREALAND_PART, AREAWATER_PART, ...
    # Detect column names defensively
    col_map = {}
    for col in xwalk.columns:
        cl = col.upper()
        if "ZCTA" in cl and ("GEOID" in cl or cl.startswith("ZCTA")):
            col_map.setdefault("zcta", col)
        elif "COUNTY" in cl and "GEOID" in cl:
            col_map.setdefault("county", col)
        elif "AREALAND_PART" in cl or (cl == "AREALAND"):
            col_map.setdefault("arealand_part", col)
        elif "AREALAND" in cl and "PART" in cl.replace("_",""):
            col_map.setdefault("arealand_part", col)

    # Fallback: take positional columns by common naming patterns
    for col in xwalk.columns:
        if col not in col_map.values():
            if "ZCTA5" in col.upper() and col_map.get("zcta") is None:
                col_map["zcta"] = col
            elif "COUNTY" in col.upper() and col_map.get("county") is None:
                col_map["county"] = col
            elif "AREALAND" in col.upper() and "PART" in col.upper() and col_map.get("arealand_part") is None:
                col_map["arealand_part"] = col

    print(f"  Column mapping: {col_map}")
    missing_cols = [k for k in ("zcta", "county") if k not in col_map]
    if missing_cols:
        print(f"  ERROR: Could not detect columns for {missing_cols}")
        print(f"  Available columns: {list(xwalk.columns)}")
        sys.exit(1)

    zcta_col   = col_map["zcta"]
    county_col = col_map["county"]

    # Standardize ZCTA to 5-digit, county to 5-digit FIPS
    xwalk["zcta5"]       = xwalk[zcta_col].str.strip().str.zfill(5)
    xwalk["county_fips"] = xwalk[county_col].str.strip().str.zfill(5)

    if "arealand_part" in col_map:
        xwalk["arealand_part"] = pd.to_numeric(xwalk[col_map["arealand_part"]], errors="coerce").fillna(0)
        # Primary county = county with largest land area intersection
        primary = (
            xwalk.sort_values("arealand_part", ascending=False)
            .groupby("zcta5", sort=False)
            .first()
            .reset_index()[["zcta5", "county_fips"]]
        )
    else:
        # No area column — take first county listed (usually the largest)
        print("  WARNING: No AREALAND_PART column found; using first county per ZCTA")
        primary = (
            xwalk.groupby("zcta5", sort=False)
            .first()
            .reset_index()[["zcta5", "county_fips"]]
        )

    primary["state_fips"] = primary["county_fips"].str[:2]

    # Log multi-county ZCTAs
    n_total   = xwalk["zcta5"].nunique()
    n_multi   = (xwalk.groupby("zcta5")["county_fips"].nunique() > 1).sum()
    print(f"  {n_total:,} unique ZCTAs; {n_multi:,} span multiple counties")

    primary.to_parquet(out, index=False)
    print(f"  Saved {len(primary):,} rows → {out.relative_to(ROOT)}")
    return primary


# ---------------------------------------------------------------------------
# 2. ZCTA centroids from TIGER shapefile
# ---------------------------------------------------------------------------

def build_zcta_centroids() -> pd.DataFrame:
    out = INTER / "zcta_centroids.parquet"
    if out.exists():
        print("  zcta_centroids.parquet already exists — loading")
        return pd.read_parquet(out)

    shp_zip = RAW / "census" / "zcta_shapefiles" / "tl_2020_us_zcta520.zip"
    require(shp_zip, "ZCTA TIGER shapefiles")

    print("  Reading ZCTA TIGER shapefile (may take a moment)...")
    gdf = gpd.read_file(f"zip://{shp_zip}")
    print(f"  Loaded {len(gdf):,} ZCTAs, CRS: {gdf.crs}")

    # Project to WGS84 for lat/lon centroids
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Compute centroids
    centroids = gdf.copy()
    centroids["centroid_lon"] = gdf.geometry.centroid.x
    centroids["centroid_lat"] = gdf.geometry.centroid.y

    # Detect ZCTA identifier column
    zcta_id_col = None
    for col in gdf.columns:
        if "ZCTA" in col.upper() and ("5" in col or "CE" in col.upper()):
            zcta_id_col = col
            break
    if zcta_id_col is None:
        zcta_id_col = gdf.columns[0]
        print(f"  WARNING: Could not detect ZCTA column; using {zcta_id_col}")

    result = (
        centroids[[zcta_id_col, "centroid_lat", "centroid_lon"]]
        .rename(columns={zcta_id_col: "zcta5"})
        .copy()
    )
    result["zcta5"] = result["zcta5"].astype(str).str.zfill(5)

    result.to_parquet(out, index=False)
    print(f"  Saved {len(result):,} ZCTA centroids → {out.relative_to(ROOT)}")
    return result


# ---------------------------------------------------------------------------
# 3. FIPS reconciliation table
# ---------------------------------------------------------------------------

def build_fips_standardized(primary: pd.DataFrame) -> pd.DataFrame:
    out = INTER / "fips_standardized.parquet"
    if out.exists():
        print("  fips_standardized.parquet already exists — loading")
        return pd.read_parquet(out)

    # Census universe: all county FIPS from the crosswalk
    census_counties = set(primary["county_fips"].unique())

    # FEMA FIPS: construct 5-digit from fipsStateCode + fipsCountyCode
    fema_path = RAW / "fema" / "disaster_declarations.parquet"
    require(fema_path, "FEMA disaster declarations (run acquire/01_fema.py)")

    fema = pd.read_parquet(fema_path, columns=["fipsStateCode", "fipsCountyCode", "disasterNumber"])
    fema["county_fips_fema"] = (
        fema["fipsStateCode"].astype(str).str.zfill(2) +
        fema["fipsCountyCode"].astype(str).str.zfill(3)
    )
    fema_counties = set(fema["county_fips_fema"].unique())

    # NOAA FIPS: CZ_TYPE == "C" county events only
    noaa_path = RAW / "noaa" / "storm_events_details.parquet"
    require(noaa_path, "NOAA storm events (run acquire/02_noaa_storm_events.py)")

    noaa = pd.read_parquet(noaa_path, columns=["STATE_FIPS", "CZ_FIPS", "CZ_TYPE"])
    noaa_county = noaa[noaa["CZ_TYPE"] == "C"].copy()
    noaa_county["county_fips_noaa"] = (
        noaa_county["STATE_FIPS"].astype(str).str.zfill(2) +
        noaa_county["CZ_FIPS"].astype(str).str.zfill(3)
    )
    noaa_counties = set(noaa_county["county_fips_noaa"].unique())

    # Build combined table
    all_counties = census_counties | fema_counties | noaa_counties
    records = []
    for fips in sorted(all_counties):
        records.append({
            "county_fips":     fips,
            "state_fips":      fips[:2],
            "in_census":       fips in census_counties,
            "in_fema":         fips in fema_counties,
            "in_noaa":         fips in noaa_counties,
        })
    df = pd.DataFrame(records)

    # Validation
    unmatched_fema = df[(df["in_fema"]) & (~df["in_census"])]["county_fips"].tolist()
    unmatched_noaa = df[(df["in_noaa"]) & (~df["in_census"])]["county_fips"].tolist()
    print(f"  FEMA counties not in Census universe: {len(unmatched_fema)}")
    print(f"  NOAA counties not in Census universe: {len(unmatched_noaa)}")
    if unmatched_fema:
        print(f"    FEMA unmatched (first 10): {unmatched_fema[:10]}")
    if unmatched_noaa:
        print(f"    NOAA unmatched (first 10): {unmatched_noaa[:10]}")
    note = (
        "NOAA zone-based events (CZ_TYPE='Z') excluded — "
        "cannot be matched to county FIPS"
    )
    print(f"  Note: {note}")

    df.to_parquet(out, index=False)
    print(f"  Saved {len(df):,} unique county FIPS → {out.relative_to(ROOT)}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Standardize Geography ===\n")

    print("1a. ZCTA → primary county")
    primary = build_zcta_county_primary()

    print("\n1b. ZCTA centroids")
    centroids = build_zcta_centroids()

    print("\n1c. FIPS reconciliation")
    fips = build_fips_standardized(primary)

    print("\n--- Verification ---")
    print(f"  zcta_county_primary:  {len(primary):,} ZCTAs")
    print(f"  zcta_centroids:       {len(centroids):,} ZCTAs")
    print(f"  fips_standardized:    {len(fips):,} county FIPS codes")
    lat_ok = centroids["centroid_lat"].between(-90, 90).all()
    lon_ok = centroids["centroid_lon"].between(-180, 180).all()
    print(f"  Centroid lat range OK: {lat_ok} | lon range OK: {lon_ok}")


if __name__ == "__main__":
    main()

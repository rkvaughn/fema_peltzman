"""
acquire/04_census.py
Download Census ACS ZCTA demographics and geographic reference files.

Outputs (data/raw/census/):
  acs_zcta_demographics.parquet     ← ~33k ZCTAs × 13 years (2011–2023)
  zcta_county_crosswalk.csv         ← ZCTA to county relationship file
  zcta_shapefiles/                  ← TIGER ZCTA shapefiles (for centroids)

Note on ACS ZCTA query:
  This script fetches the full national ZCTA universe in one call per year
  (no state filter). This differs from the census_api.py utility which
  fetches tracts by state. The ZCTA geography is: "zip code tabulation area:*"

Usage:
  CENSUS_API_KEY=your_key python acquire/04_census.py
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW_CENSUS = ROOT / "data" / "raw" / "census"
RAW_CENSUS.mkdir(parents=True, exist_ok=True)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
if not CENSUS_API_KEY:
    print("WARNING: CENSUS_API_KEY not set. Requests will be rate-limited to 500/day.")
    print("  Set via: export CENSUS_API_KEY=your_key  or add to .env")

# ACS variables per the project plan (fema-peltzman-project-plan-v2.md §3A)
ACS_VARS = [
    "B01003_001E",  # total population
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25003_001E",  # housing tenure total
    "B25003_002E",  # owner-occupied
    "B25003_003E",  # renter-occupied
    "B25002_001E",  # occupancy status total
    "B25002_002E",  # occupied
    "B25002_003E",  # vacant
    "B01002_001E",  # median age
    "B25034_001E",  # year built total
    "B25034_010E",  # built 1939 or earlier (housing age proxy)
    "B25024_002E",  # 1-unit detached (SFH)
    "B25024_003E",  # 1-unit attached
    "B02001_001E",  # race total
    "B02001_002E",  # white alone
    "B02001_003E",  # Black alone
    "B03001_001E",  # hispanic origin total
    "B03001_003E",  # hispanic
    "B25001_001E",  # total housing units
]

ACS_YEARS = list(range(2011, 2024))  # 2011–2023 (5-year ACS vintages)


# ---------------------------------------------------------------------------
# ACS ZCTA download
# ---------------------------------------------------------------------------

def download_acs_zcta() -> None:
    out = RAW_CENSUS / "acs_zcta_demographics.parquet"
    if out.exists():
        df_existing = pd.read_parquet(out, columns=["acs_year"])
        years_present = sorted(df_existing["acs_year"].unique())
        if len(years_present) >= len(ACS_YEARS):
            print(f"  acs_zcta_demographics.parquet already exists ({len(df_existing):,} rows) — skipping")
            return
        else:
            years_missing = [y for y in ACS_YEARS if y not in years_present]
            print(f"  Partial download detected. Missing years: {years_missing}")

    print(f"Downloading ACS 5-year ZCTA data for {ACS_YEARS[0]}–{ACS_YEARS[-1]}...")
    var_string = ",".join(ACS_VARS)
    all_dfs = []

    for year in ACS_YEARS:
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        params = {
            "get":  f"NAME,{var_string}",
            "for":  "zip code tabulation area:*",
        }
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY

        print(f"  {year}...", end=" ")
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data[1:], columns=data[0])
                df["acs_year"] = year
                # Convert numeric columns from strings
                for col in ACS_VARS:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                all_dfs.append(df)
                print(f"{len(df):,} ZCTAs")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAILED — {e}")
                else:
                    time.sleep(5 * (attempt + 1))

        time.sleep(1.0)

    if not all_dfs:
        print("ERROR: No ACS data downloaded")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(out, index=False)
    print(f"\nSaved {len(combined):,} ZCTA-year observations → {out.relative_to(ROOT)}")

    # Validation
    years_present = sorted(combined["acs_year"].unique())
    years_missing = [y for y in ACS_YEARS if y not in years_present]
    if years_missing:
        print(f"WARNING: Missing years: {years_missing}")
    else:
        print(f"All {len(ACS_YEARS)} years present")
    print(f"Unique ZCTAs per year (approx): {combined.groupby('acs_year')['zip code tabulation area'].nunique().mean():.0f}")


# ---------------------------------------------------------------------------
# ZCTA-to-county relationship file
# ---------------------------------------------------------------------------

def download_zcta_county_crosswalk() -> None:
    out = RAW_CENSUS / "zcta_county_crosswalk.csv"
    if out.exists():
        print(f"  zcta_county_crosswalk.csv already exists — skipping")
        return

    # 2020 ZCTA5 to 2020 county relationship file (national)
    # Source: Census Bureau relationship files page
    url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt"
    print(f"Downloading ZCTA-to-County crosswalk...")
    try:
        df = pd.read_csv(url, sep="|", dtype=str, low_memory=False)
        df.to_csv(out, index=False)
        print(f"  Saved {len(df):,} ZCTA-county pairs → {out.relative_to(ROOT)}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Manual download from: https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html")
        print("  Save to: data/raw/census/zcta_county_crosswalk.csv")


# ---------------------------------------------------------------------------
# ZCTA TIGER shapefiles (for ZIP centroids)
# ---------------------------------------------------------------------------

def download_zcta_shapefiles() -> None:
    """
    Download Census TIGER ZCTA shapefiles for computing ZIP centroids.
    These are used by preprocessing/04_hazard_intensity.py for spatial
    interpolation of USGS high water marks and HURDAT2 wind field.

    We download the 2020 ZCTA shapefile (most current stable vintage).
    """
    shp_dir = RAW_CENSUS / "zcta_shapefiles"
    shp_dir.mkdir(exist_ok=True)

    out_zip = shp_dir / "tl_2020_us_zcta520.zip"
    if out_zip.exists():
        print(f"  ZCTA shapefile already present — skipping")
        return

    url = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
    print(f"Downloading ZCTA TIGER shapefile ({url})...")
    print("  (Large file — ~500 MB)")

    try:
        with requests.get(url, timeout=600, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(out_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"  {pct:.1f}% ({downloaded / 1024**2:.0f} MB)", end="\r")
        print(f"\n  Downloaded {out_zip.stat().st_size / 1024**2:.0f} MB → {out_zip.relative_to(ROOT)}")

        # Verify it's a valid zip
        with zipfile.ZipFile(out_zip) as zf:
            names = zf.namelist()
            print(f"  Contains: {', '.join(names[:5])} ...")

    except Exception as e:
        print(f"\n  FAILED: {e}")
        print("  Manual download: https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip")
        print(f"  Save to: {out_zip.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== ACS ZCTA Demographics ===")
    download_acs_zcta()

    print("\n=== ZCTA-to-County Crosswalk ===")
    download_zcta_county_crosswalk()

    print("\n=== ZCTA TIGER Shapefiles ===")
    download_zcta_shapefiles()

    print("\n--- Summary ---")
    checks = [
        RAW_CENSUS / "acs_zcta_demographics.parquet",
        RAW_CENSUS / "zcta_county_crosswalk.csv",
        RAW_CENSUS / "zcta_shapefiles" / "tl_2020_us_zcta520.zip",
    ]
    for path in checks:
        exists = path.exists()
        mark = "✓" if exists else "✗"
        size = f"({path.stat().st_size / 1024**2:.0f} MB)" if exists else ""
        print(f"  {mark}  {path.relative_to(ROOT)} {size}")


if __name__ == "__main__":
    main()

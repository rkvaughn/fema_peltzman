"""
acquire/03_hazard_point_sources.py
Download three hazard data sources:
  (A) USGS Short-Term Network (STN) — flood high water marks
  (B) NOAA NHC HURDAT2 — hurricane tracks
  (C) NWS warnings via Iowa Environmental Mesonet (IEM) — CSV + shapefiles

Outputs:
  data/raw/usgs/stn_events.parquet
  data/raw/usgs/high_water_marks.parquet
  data/raw/usgs/stn_sites.parquet
  data/raw/nhc/hurdat2.parquet
  data/raw/nws/warnings_{year}.csv          (2000–2025)
  data/raw/nws/warnings_shp_{year}.zip      (IEM shapefiles — needed for spatial joins)
  data/raw/nws/nws_warnings.parquet         (filtered CSV combined)

IMPORTANT: Both CSV and shapefile formats are downloaded from IEM.
The shapefiles are required by preprocessing/04_hazard_intensity.py for
spatial joins. Do not skip the shapefile download.

Usage:
  python acquire/03_hazard_point_sources.py [--usgs] [--hurdat] [--nws]
  (no flags = run all three)
"""

import os
import sys
import time
import glob
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
RAW_USGS = ROOT / "data" / "raw" / "usgs"
RAW_NHC  = ROOT / "data" / "raw" / "nhc"
RAW_NWS  = ROOT / "data" / "raw" / "nws"

for d in [RAW_USGS, RAW_NHC, RAW_NWS]:
    d.mkdir(parents=True, exist_ok=True)

NWS_YEARS = range(2000, 2026)
# Phenomena relevant to this analysis: flood, hurricane, tornado, severe storm
NWS_PHENOMS = {"FF", "FL", "HU", "TO", "SV"}


# ---------------------------------------------------------------------------
# (A) USGS STN — High Water Marks
# ---------------------------------------------------------------------------

def download_usgs_stn():
    STN_BASE = "https://stn.wim.usgs.gov/STNServices"
    endpoints = {
        "stn_events":       f"{STN_BASE}/Events.json",
        "high_water_marks": f"{STN_BASE}/HWMs.json",
        "stn_sites":        f"{STN_BASE}/Sites.json",
    }

    for name, url in endpoints.items():
        out = RAW_USGS / f"{name}.parquet"
        if out.exists():
            print(f"  {name}.parquet already exists — skipping")
            continue

        print(f"  Downloading USGS STN {name}...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            records = resp.json()
            df = pd.DataFrame(records)
            df.to_parquet(out, index=False)
            print(f"  Saved {len(df):,} rows → {out.relative_to(ROOT)}")
        except Exception as e:
            print(f"  FAILED {name}: {e}")


# ---------------------------------------------------------------------------
# (B) NOAA NHC HURDAT2
# ---------------------------------------------------------------------------

def download_hurdat2():
    out_txt = RAW_NHC / "hurdat2.txt"
    out_parquet = RAW_NHC / "hurdat2.parquet"

    if out_parquet.exists():
        print("  hurdat2.parquet already exists — skipping")
        return

    url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
    print(f"  Downloading HURDAT2 from {url}...")

    if not out_txt.exists():
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                out_txt.write_bytes(resp.content)
                print(f"  Saved raw HURDAT2 ({out_txt.stat().st_size / 1024:.0f} KB)")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED downloading HURDAT2: {e}")
                    return
                time.sleep(5)

    # Parse fixed-width HURDAT2 format
    print("  Parsing HURDAT2...")
    records = []
    current_storm = None

    with open(out_txt) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) == 4:
                # Header line: AL012000, ALBERTO, 11,
                current_storm = {
                    "storm_id":   parts[0].strip(),
                    "storm_name": parts[1].strip(),
                    "n_entries":  int(parts[2].strip()) if parts[2].strip().isdigit() else None,
                }
            elif len(parts) >= 8 and current_storm is not None:
                # Track point line
                def safe_int(s):
                    s = s.strip()
                    return int(s) if s.lstrip("-").isdigit() else None

                def parse_coord(s):
                    s = s.strip()
                    val = float(s[:-1])
                    return -val if s[-1] in ("S", "W") else val

                try:
                    records.append({
                        "storm_id":        current_storm["storm_id"],
                        "storm_name":      current_storm["storm_name"],
                        "date":            parts[0].strip(),
                        "time_utc":        parts[1].strip(),
                        "record_id":       parts[2].strip(),
                        "status":          parts[3].strip(),
                        "lat":             parse_coord(parts[4]),
                        "lon":             parse_coord(parts[5]),
                        "max_wind_kt":     safe_int(parts[6]),
                        "min_pressure_mb": safe_int(parts[7]),
                        # Extended track wind radii if present
                        "r34_ne": safe_int(parts[8])  if len(parts) > 8  else None,
                        "r34_se": safe_int(parts[9])  if len(parts) > 9  else None,
                        "r34_sw": safe_int(parts[10]) if len(parts) > 10 else None,
                        "r34_nw": safe_int(parts[11]) if len(parts) > 11 else None,
                        "r50_ne": safe_int(parts[12]) if len(parts) > 12 else None,
                        "r50_se": safe_int(parts[13]) if len(parts) > 13 else None,
                        "r50_sw": safe_int(parts[14]) if len(parts) > 14 else None,
                        "r50_nw": safe_int(parts[15]) if len(parts) > 15 else None,
                        "r64_ne": safe_int(parts[16]) if len(parts) > 16 else None,
                        "r64_se": safe_int(parts[17]) if len(parts) > 17 else None,
                        "r64_sw": safe_int(parts[18]) if len(parts) > 18 else None,
                        "r64_nw": safe_int(parts[19]) if len(parts) > 19 else None,
                    })
                except Exception:
                    pass  # malformed line — skip

    df = pd.DataFrame(records)
    df.to_parquet(out_parquet, index=False)
    print(f"  Parsed {len(df):,} track points, {df['storm_id'].nunique():,} storms")
    print(f"  Saved → {out_parquet.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# (C) NWS Warnings via IEM
# ---------------------------------------------------------------------------

def download_nws_warnings():
    """
    Download NWS watch/warn/advisory records from Iowa Environmental Mesonet.

    Two formats per year:
      1. CSV — tabular data for filtering and analysis
      2. Shapefile ZIP — polygon geometry required for spatial joins in
         preprocessing/04_hazard_intensity.py

    The shapefile download is NOT optional; it is required for computing
    warning lead time per county.
    """
    csv_base = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
    shp_base = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"

    csv_missing = []
    shp_missing = []

    for year in NWS_YEARS:
        csv_out = RAW_NWS / f"warnings_{year}.csv"
        shp_out = RAW_NWS / f"warnings_shp_{year}.zip"

        if not csv_out.exists():
            csv_missing.append(year)
        if not shp_out.exists():
            shp_missing.append(year)

    if not csv_missing and not shp_missing:
        print("  All NWS warning files already present — skipping download")
    else:
        # Download CSV files
        if csv_missing:
            print(f"  Downloading NWS warning CSVs for {len(csv_missing)} years...")
            for year in csv_missing:
                _download_nws_year_csv(year, csv_base)
                time.sleep(2)

        # Download shapefiles
        if shp_missing:
            print(f"\n  Downloading NWS warning shapefiles for {len(shp_missing)} years...")
            print("  (Required for spatial joins in preprocessing/04_hazard_intensity.py)")
            for year in shp_missing:
                _download_nws_year_shp(year, shp_base)
                time.sleep(2)

    # Combine CSV files into filtered parquet
    _combine_nws_csvs()


def _download_nws_year_csv(year: int, base_url: str) -> None:
    out = RAW_NWS / f"warnings_{year}.csv"
    params = {
        "year1": year, "month1": 1,  "day1": 1,
        "year2": year, "month2": 12, "day2": 31,
        "fmt": "csv",
    }
    print(f"  CSV {year}...", end=" ")
    try:
        resp = requests.get(base_url, params=params, timeout=300)
        resp.raise_for_status()
        out.write_text(resp.text, encoding="utf-8")
        lines = resp.text.count("\n")
        print(f"{lines:,} lines")
    except Exception as e:
        print(f"FAILED — {e}")


def _download_nws_year_shp(year: int, base_url: str) -> None:
    out = RAW_NWS / f"warnings_shp_{year}.zip"
    params = {
        "year1": year, "month1": 1,  "day1": 1,
        "year2": year, "month2": 12, "day2": 31,
        "fmt": "shp",
    }
    print(f"  SHP {year}...", end=" ")
    try:
        resp = requests.get(base_url, params=params, timeout=600, stream=True)
        resp.raise_for_status()
        content = b"".join(resp.iter_content(65536))
        out.write_bytes(content)
        print(f"{len(content) / 1024:.0f} KB")
    except Exception as e:
        print(f"FAILED — {e}")


def _combine_nws_csvs() -> None:
    """
    Read all annual CSV files, filter to relevant phenomena before concat
    to avoid OOM on large datasets, save combined parquet.
    """
    out = RAW_NWS / "nws_warnings.parquet"
    if out.exists():
        print("\n  nws_warnings.parquet already exists — skipping combine")
        return

    csv_files = sorted(RAW_NWS.glob("warnings_*.csv"))
    if not csv_files:
        print("  WARNING: No NWS CSV files found to combine")
        return

    print(f"\n  Combining {len(csv_files)} NWS CSV files (filtering to relevant phenomena)...")
    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            # Filter before concat to avoid OOM
            if "PHENOM" in df.columns and "SIG" in df.columns:
                df = df[df["PHENOM"].isin(NWS_PHENOMS) & (df["SIG"] == "W")]
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {path.name}: {e}")

    if not dfs:
        print("  ERROR: No CSV files could be read")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(out, index=False)
    print(f"  Combined {len(combined):,} relevant NWS warning records")
    print(f"  Saved → {out.relative_to(ROOT)}")

    if "PHENOM" in combined.columns:
        print("\n  Phenomenon distribution (filtered):")
        for ph, cnt in combined["PHENOM"].value_counts().items():
            print(f"    {ph}: {cnt:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download hazard point source data")
    parser.add_argument("--usgs",   action="store_true", help="USGS STN only")
    parser.add_argument("--hurdat", action="store_true", help="HURDAT2 only")
    parser.add_argument("--nws",    action="store_true", help="NWS warnings only")
    args = parser.parse_args()

    run_all = not (args.usgs or args.hurdat or args.nws)

    if run_all or args.usgs:
        print("=== USGS STN ===")
        download_usgs_stn()

    if run_all or args.hurdat:
        print("\n=== NOAA HURDAT2 ===")
        download_hurdat2()

    if run_all or args.nws:
        print("\n=== NWS Warnings (IEM) ===")
        download_nws_warnings()

    print("\n--- Summary ---")
    checks = [
        RAW_USGS / "stn_events.parquet",
        RAW_USGS / "high_water_marks.parquet",
        RAW_USGS / "stn_sites.parquet",
        RAW_NHC  / "hurdat2.parquet",
        RAW_NWS  / "nws_warnings.parquet",
    ]
    for path in checks:
        mark = "✓" if path.exists() else "✗"
        print(f"  {mark}  {path.relative_to(ROOT)}")

    shp_count = len(list(RAW_NWS.glob("warnings_shp_*.zip")))
    csv_count = len(list(RAW_NWS.glob("warnings_*.csv")))
    print(f"  NWS CSVs: {csv_count}/26 years   Shapefiles: {shp_count}/26 years")
    if shp_count < 20:
        print("  WARNING: Fewer than 20 shapefile years present.")
        print("  preprocessing/04_hazard_intensity.py requires shapefiles for spatial joins.")


if __name__ == "__main__":
    main()

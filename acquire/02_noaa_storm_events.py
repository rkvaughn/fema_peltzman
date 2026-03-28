"""
acquire/02_noaa_storm_events.py
Download NOAA NCEI Storm Events database (2000–2024).

Outputs (data/raw/noaa/):
  StormEvents_details_*.csv.gz   ← raw annual files (kept for reference)
  storm_events_details.parquet   ← combined, all years

Usage:
  python acquire/02_noaa_storm_events.py

Validation: prints year range and event type distribution after download.
A missing year is a silent failure — check the output carefully.
"""

import os
import sys
import time
import glob
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_NOAA = ROOT / "data" / "raw" / "noaa"
RAW_NOAA.mkdir(parents=True, exist_ok=True)

NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEARS = range(2000, 2025)


def list_remote_files() -> list[str]:
    """Scrape the NCEI directory listing for StormEvents_details files."""
    resp = requests.get(NCEI_BASE, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [
        a["href"] for a in soup.find_all("a", href=True)
        if "StormEvents_details" in a["href"] and a["href"].endswith(".csv.gz")
    ]
    return sorted(links)


def download_annual_files(remote_files: list[str]) -> list[Path]:
    """Download each annual file if not already present."""
    downloaded = []
    for fname in remote_files:
        # Extract year from filename: StormEvents_details-ftp_v1.0_dYYYY*.csv.gz
        try:
            year_str = [p for p in fname.split("_") if p.startswith("d")][0][1:5]
            year = int(year_str)
        except (IndexError, ValueError):
            continue

        if year not in YEARS:
            continue

        out = RAW_NOAA / fname.split("/")[-1]
        if out.exists():
            print(f"  {year}: already present — skipping")
            downloaded.append(out)
            continue

        url = NCEI_BASE + fname.split("/")[-1]
        print(f"  {year}: downloading {url}")
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                print(f"  {year}: saved ({out.stat().st_size / 1024:.0f} KB)")
                downloaded.append(out)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  {year}: FAILED — {e}")
                else:
                    time.sleep(5 * (attempt + 1))

        time.sleep(0.5)

    return downloaded


def combine_to_parquet(gz_files: list[Path]) -> pd.DataFrame:
    """Read all .csv.gz files, filter to YEARS, combine."""
    dfs = []
    for path in sorted(gz_files):
        try:
            df = pd.read_csv(path, low_memory=False, compression="gzip")
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {path.name}: {e}")

    if not dfs:
        print("ERROR: No files to combine.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    # Validate year coverage
    if "YEAR" in combined.columns:
        years_present = sorted(combined["YEAR"].unique())
        years_missing = [y for y in YEARS if y not in years_present]
        print(f"\n  Year range: {min(years_present)} – {max(years_present)}")
        if years_missing:
            print(f"  WARNING: Missing years: {years_missing}")
        else:
            print(f"  All years {min(YEARS)}–{max(YEARS)} present")

        # Event type distribution
        if "EVENT_TYPE" in combined.columns:
            print("\n  Top event types:")
            top = combined["EVENT_TYPE"].value_counts().head(10)
            for etype, cnt in top.items():
                print(f"    {cnt:>8,}  {etype}")

    return combined


def main():
    out_parquet = RAW_NOAA / "storm_events_details.parquet"
    if out_parquet.exists():
        print("storm_events_details.parquet already exists.")
        df = pd.read_parquet(out_parquet)
        print(f"  {len(df):,} records, columns: {list(df.columns[:8])} ...")
        return

    print("Fetching NCEI directory listing...")
    remote_files = list_remote_files()
    print(f"  Found {len(remote_files)} StormEvents_details files on NCEI")

    gz_files_on_disk = sorted(RAW_NOAA.glob("StormEvents_details*.csv.gz"))
    if len(gz_files_on_disk) >= len([f for f in remote_files
                                     if any(str(y) in f for y in YEARS)]):
        print("  All annual files already downloaded — combining...")
    else:
        print("\nDownloading annual storm events files...")
        gz_files_on_disk = download_annual_files(remote_files)

    print("\nCombining into parquet...")
    combined = combine_to_parquet(gz_files_on_disk)

    combined.to_parquet(out_parquet, index=False)
    print(f"\nSaved {len(combined):,} rows → {out_parquet.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

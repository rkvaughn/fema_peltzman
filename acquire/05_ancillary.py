"""
acquire/05_ancillary.py
Download ancillary datasets:
  (A) Treasury Total Taxable Resources (TTR) — state fiscal capacity
  (B) Election data — MIT Election Lab presidential returns, Klarner governor party
  (C) FEMA IA threshold stub — UNVERIFIED values from project plan

Outputs (data/raw/):
  treasury/ttr_raw.xlsx                 ← raw Treasury download (if available)
  treasury/ttr_raw_parsed.csv           ← auto-parsed (needs manual cleanup)
  elections/county_pres_returns.csv     ← MIT Election Lab county presidential returns
  elections/klarner_state_politics.csv  ← Klarner governor/legislature party data
  fema/ia_thresholds_UNVERIFIED.csv     ← threshold stub; verified=False on all rows

IMPORTANT — THRESHOLD STUB:
  The ia_thresholds_UNVERIFIED.csv file contains approximate values from the
  project research plan. ALL rows have verified=False. The preprocessing
  pipeline (04b_construct_ttr_adjustment.py) will refuse to run until each
  value is verified against the Federal Register and verified=True is set.
  DO NOT set verified=True without checking the Federal Register notices
  for 44 CFR 206.48.

Usage:
  python acquire/05_ancillary.py
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_TREASURY  = ROOT / "data" / "raw" / "treasury"
RAW_ELECTIONS = ROOT / "data" / "raw" / "elections"
RAW_FEMA      = ROOT / "data" / "raw" / "fema"

for d in [RAW_TREASURY, RAW_ELECTIONS, RAW_FEMA]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# (A) Treasury TTR
# ---------------------------------------------------------------------------

def download_ttr() -> None:
    out_xlsx = RAW_TREASURY / "ttr_raw.xlsx"
    out_csv  = RAW_TREASURY / "ttr_raw_parsed.csv"

    if out_xlsx.exists():
        print("  ttr_raw.xlsx already exists — skipping download")
        _parse_ttr_excel(out_xlsx, out_csv)
        return

    # Treasury TTR page — URL changes with each update; try known patterns
    ttr_urls = [
        "https://home.treasury.gov/system/files/131/TTR-Table1.xlsx",
        "https://home.treasury.gov/system/files/131/Total-Taxable-Resources.xlsx",
    ]

    print("Downloading Treasury TTR data...")
    downloaded = False
    for url in ttr_urls:
        print(f"  Trying: {url}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_xlsx.write_bytes(resp.content)
            print(f"  Downloaded ({out_xlsx.stat().st_size / 1024:.0f} KB)")
            downloaded = True
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if not downloaded:
        print("\n  Could not auto-download TTR data.")
        print("  Manual download required:")
        print("  URL: https://home.treasury.gov/policy-issues/economic-policy/total-taxable-resources")
        print(f"  Save to: {out_xlsx.relative_to(ROOT)}")
        print("\n  After download, re-run this script to parse.")
        return

    _parse_ttr_excel(out_xlsx, out_csv)


def _parse_ttr_excel(xlsx_path: Path, csv_path: Path) -> None:
    """
    Attempt to parse the Treasury TTR Excel file.
    TTR files have complex multi-row headers; this is a best-effort parse.
    Manual inspection of ttr_raw_parsed.csv is required to confirm column mapping.
    """
    if csv_path.exists():
        print("  ttr_raw_parsed.csv already exists — skipping parse")
        return

    print("  Parsing TTR Excel (complex layout — manual review required)...")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, header=None)
        df.to_csv(csv_path, index=False)
        print(f"  Parsed: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Saved → {csv_path.relative_to(ROOT)}")
        print("\n  NEXT STEP: Inspect ttr_raw_parsed.csv and manually produce")
        print("  treasury/ttr_clean.csv with columns:")
        print("    state_name, state_fips, fiscal_year, ttr_total_millions,")
        print("    ttr_per_capita, ttr_pct_us_avg")
        print("  Required for preprocessing/04b_construct_ttr_adjustment.py")
    except Exception as e:
        print(f"  Parse failed: {e}")
        print("  Manual parsing required.")


# ---------------------------------------------------------------------------
# (B) Election Data
# ---------------------------------------------------------------------------

def download_election_data() -> None:
    _download_mit_presidential_returns()
    _download_klarner_governor_party()


def _download_mit_presidential_returns() -> None:
    out = RAW_ELECTIONS / "county_pres_returns.csv"
    if out.exists():
        print("  county_pres_returns.csv already exists — skipping")
        return

    # MIT Election Data + Science Lab, Harvard Dataverse
    # County Presidential Election Returns 2000-2020
    # Direct CSV from their GitHub repository
    url = "https://raw.githubusercontent.com/MEDSL/county-returns/main/countypres_2000-2020.csv"
    print(f"Downloading MIT Election Lab county presidential returns...")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out.write_text(resp.text, encoding="utf-8")
        df = pd.read_csv(out)
        print(f"  Saved {len(df):,} rows → {out.relative_to(ROOT)}")
        print(f"  Years: {sorted(df['year'].unique()) if 'year' in df.columns else 'unknown'}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Alternative: Download from Harvard Dataverse:")
        print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ")
        print(f"  Save CSV to: {out.relative_to(ROOT)}")


def _download_klarner_governor_party() -> None:
    out = RAW_ELECTIONS / "klarner_state_politics.csv"
    if out.exists():
        print("  klarner_state_politics.csv already exists — skipping")
        return

    # Klarner Politics dataset — state government partisan composition
    # Available from Harvard Dataverse
    # Try the direct CSV download URL (may require manual download if URL changes)
    urls = [
        "https://raw.githubusercontent.com/CenterForPeaceAndSecurityStudies/ICBEdataset/master/replication_data/klarner_partisan_state_control.csv",
    ]
    print("Downloading Klarner state politics data (governor party)...")
    downloaded = False
    for url in urls:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out.write_text(resp.text, encoding="utf-8")
            df = pd.read_csv(out)
            print(f"  Saved {len(df):,} rows → {out.relative_to(ROOT)}")
            downloaded = True
            break
        except Exception as e:
            print(f"  URL failed: {e}")

    if not downloaded:
        print("  Could not auto-download Klarner data.")
        print("  Manual download from Harvard Dataverse:")
        print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SAWKXB")
        print(f"  Save to: {out.relative_to(ROOT)}")
        print("  Key columns needed: state, year, governor_party (R/D)")


# ---------------------------------------------------------------------------
# (C) FEMA IA Threshold Stub
# ---------------------------------------------------------------------------

def create_threshold_stub() -> None:
    """
    Create the FEMA IA threshold stub file with UNVERIFIED values from the
    project research plan (fema-peltzman-project-plan-v2.md §4B).

    ALL rows have verified=False. The preprocessing pipeline will refuse to
    run 04b_construct_ttr_adjustment.py until values are verified against
    Federal Register notices (44 CFR 206.48).

    Confidence in these values: 6/10 per project plan.
    DO NOT use in analysis without verification.

    Values from: fema-peltzman-project-plan-v2.md §4B (confirmed pre-specified
    in the project research plan — awaiting PI verification against Federal Register).
    """
    out = RAW_FEMA / "ia_thresholds_UNVERIFIED.csv"
    if out.exists():
        print("  ia_thresholds_UNVERIFIED.csv already exists — skipping")
        return

    # Values pre-specified in project research plan §4B.
    # Source: fema-peltzman-project-plan-v2.md, table in §4B
    # Confidence: 6/10. MUST verify all against Federal Register before use.
    # The county threshold was restructured in 2008 and again in 2016.
    thresholds = [
        # year, state_threshold ($/capita), county_threshold ($/capita)
        (2000, 1.07, 3.20),
        (2001, 1.09, 3.27),
        (2002, 1.11, 3.33),
        (2003, 1.12, 3.36),
        (2004, 1.14, 3.42),
        (2005, 1.16, 3.48),
        (2006, 1.19, 3.57),
        (2007, 1.22, 3.66),
        (2008, 1.29, 3.50),  # restructured 2008
        (2009, 1.31, 3.56),
        (2010, 1.35, 3.61),
        (2011, 1.37, 3.64),
        (2012, 1.39, 3.68),
        (2013, 1.41, 3.72),
        (2014, 1.43, 3.78),
        (2015, 1.46, 3.86),
        (2016, 1.47, 3.50),  # restructured 2016
        (2017, 1.50, 3.57),
        (2018, 1.52, 3.62),
        (2019, 1.55, 3.68),
        (2020, 1.57, 3.72),
        (2021, 1.59, 3.78),
        (2022, 1.63, 3.86),
        (2023, 1.68, 3.97),
        (2024, 1.74, 4.12),
        (2025, 1.79, 4.24),
    ]

    df = pd.DataFrame(thresholds, columns=["year", "state_threshold", "county_threshold"])
    df["verified"] = False
    df["source"] = "fema-peltzman-project-plan-v2.md §4B (pre-specified, unverified)"
    df["notes"] = ""
    df.loc[df["year"] == 2008, "notes"] = "threshold restructured 2008"
    df.loc[df["year"] == 2016, "notes"] = "threshold restructured 2016"

    df.to_csv(out, index=False)
    print(f"  Created threshold stub: {len(df)} years → {out.relative_to(ROOT)}")
    print()
    print("  *** ACTION REQUIRED ***")
    print("  All threshold values are UNVERIFIED (verified=False).")
    print("  Before running preprocessing/04b_construct_ttr_adjustment.py:")
    print("  1. Look up each year's threshold in the Federal Register (44 CFR 206.48)")
    print("  2. Correct any values that differ from the published notice")
    print("  3. Set verified=True for each confirmed row")
    print("  4. Add the Federal Register citation to the 'source' column")
    print()
    print("  Federal Register search: https://www.federalregister.gov/")
    print("  Search terms: 'FEMA Individual Assistance thresholds' '44 CFR 206.48'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Treasury TTR ===")
    download_ttr()

    print("\n=== Election Data ===")
    download_election_data()

    print("\n=== FEMA IA Threshold Stub ===")
    create_threshold_stub()

    print("\n--- Summary ---")
    checks = {
        RAW_TREASURY  / "ttr_raw.xlsx":                "TTR raw Excel",
        RAW_TREASURY  / "ttr_raw_parsed.csv":          "TTR parsed CSV",
        RAW_ELECTIONS / "county_pres_returns.csv":     "MIT presidential returns",
        RAW_ELECTIONS / "klarner_state_politics.csv":  "Klarner governor party",
        RAW_FEMA      / "ia_thresholds_UNVERIFIED.csv": "IA threshold stub",
    }
    for path, label in checks.items():
        mark = "✓" if path.exists() else "✗  (manual download needed)"
        print(f"  {mark}  {label}")

    print()
    print("Next steps requiring manual action:")
    print("  1. Inspect treasury/ttr_raw_parsed.csv → produce treasury/ttr_clean.csv")
    print("  2. Verify IA threshold values against Federal Register")
    print("  3. If MIT/Klarner downloads failed, download manually (see URLs above)")


if __name__ == "__main__":
    main()

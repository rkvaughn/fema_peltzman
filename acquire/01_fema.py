"""
acquire/01_fema.py
Download core FEMA datasets from OpenFEMA REST API.

Outputs (data/raw/fema/):
  disaster_declarations.parquet
  housing_assistance_owners.parquet
  housing_assistance_renters.parquet
  ia_registrants_large_disasters.parquet
  nfip_claims.parquet          ← largest; checkpointed every 100k rows
  pa_applicants.parquet
  pa_projects.parquet

Usage:
  cd ~/Projects/fema_peltzman
  source .venv/bin/activate
  python acquire/01_fema.py
"""

import os
import sys
import time
import glob
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FEMA = ROOT / "data" / "raw" / "fema"
RAW_FEMA.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core pagination helper
# ---------------------------------------------------------------------------

def openfema_paginate(
    endpoint: str,
    data_key: str,
    *,
    filters: str = "",
    batch_size: int = 1000,
    sleep: float = 0.4,
    timeout: int = 120,
    max_records: int | None = None,
) -> list[dict]:
    """
    Paginate an OpenFEMA REST endpoint using OData $top/$skip.

    Args:
        endpoint:   Full URL, e.g. https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries
        data_key:   Key in the JSON response that holds the records list.
        filters:    Optional OData $filter string, e.g. "$filter=incidentType eq 'Flood'"
        batch_size: Records per page (max 1000).
        sleep:      Seconds between requests.
        timeout:    Requests timeout in seconds.
        max_records: Stop after this many records (for testing).
    """
    all_records = []
    skip = 0

    # Get total count first for progress reporting
    count_url = f"{endpoint}?$count=true&$top=1"
    if filters:
        count_url += f"&{filters}"
    try:
        total = requests.get(count_url, timeout=30).json()["metadata"]["count"]
        print(f"  Total records available: {total:,}")
    except Exception:
        total = None
        print("  (Could not retrieve total count)")

    while True:
        params_str = f"?$top={batch_size}&$skip={skip}&$format=json"
        if filters:
            params_str += f"&{filters}"
        url = endpoint + params_str

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED after 3 attempts at skip={skip}: {e}")
                    return all_records
                time.sleep(5 * (attempt + 1))

        batch = resp.json().get(data_key, [])
        if not batch:
            break

        all_records.extend(batch)
        skip += len(batch)

        pct = f" ({skip/total*100:.1f}%)" if total else ""
        print(f"  Fetched {skip:,} records{pct}", end="\r")

        if max_records and skip >= max_records:
            break
        if len(batch) < batch_size:
            break

        time.sleep(sleep)

    print(f"  Fetched {len(all_records):,} total records        ")
    return all_records


def save_parquet(records: list[dict], path: Path, label: str) -> None:
    if not records:
        print(f"  WARNING: No records for {label} — skipping save")
        return
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df):,} rows → {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Dataset 1: Disaster Declarations Summaries v2
# ---------------------------------------------------------------------------

def download_disaster_declarations():
    out = RAW_FEMA / "disaster_declarations.parquet"
    if out.exists():
        print("  disaster_declarations.parquet already exists — skipping")
        return

    print("Downloading Disaster Declarations Summaries...")
    records = openfema_paginate(
        "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries",
        data_key="DisasterDeclarationsSummaries",
    )
    save_parquet(records, out, "disaster_declarations")


# ---------------------------------------------------------------------------
# Dataset 2+3: Housing Assistance Owners/Renters v1 (ZIP-level IA outcomes)
# ---------------------------------------------------------------------------

def download_housing_assistance():
    datasets = {
        "housing_assistance_owners": (
            "https://www.fema.gov/api/open/v1/HousingAssistanceOwners",
            "HousingAssistanceOwners",
        ),
        "housing_assistance_renters": (
            "https://www.fema.gov/api/open/v1/HousingAssistanceRenters",
            "HousingAssistanceRenters",
        ),
    }
    for name, (endpoint, data_key) in datasets.items():
        out = RAW_FEMA / f"{name}.parquet"
        if out.exists():
            print(f"  {name}.parquet already exists — skipping")
            continue
        print(f"Downloading {name}...")
        records = openfema_paginate(endpoint, data_key=data_key)
        save_parquet(records, out, name)


# ---------------------------------------------------------------------------
# Dataset 4: IA Registrants (applicant-level, large disasters)
# ---------------------------------------------------------------------------

def download_ia_registrants():
    out = RAW_FEMA / "ia_registrants_large_disasters.parquet"
    if out.exists():
        print("  ia_registrants_large_disasters.parquet already exists — skipping")
        return

    print("Downloading IA Registrants (Large Disasters)...")
    records = openfema_paginate(
        "https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters",
        data_key="IndividualAssistanceHousingRegistrantsLargeDisasters",
        sleep=0.5,
    )
    save_parquet(records, out, "ia_registrants")


# ---------------------------------------------------------------------------
# Dataset 5: NFIP Claims (largest dataset — checkpointed)
# ---------------------------------------------------------------------------

def download_nfip_claims():
    out = RAW_FEMA / "nfip_claims.parquet"
    if out.exists():
        print("  nfip_claims.parquet already exists — skipping")
        return

    checkpoint_dir = RAW_FEMA / "nfip_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("Downloading NFIP Claims (large dataset — may take several hours)...")
    endpoint = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
    data_key = "FimaNfipClaims"

    # Determine starting skip from existing checkpoints
    existing = sorted(glob.glob(str(checkpoint_dir / "checkpoint_*.parquet")))
    skip = 0
    all_records = []

    if existing:
        for cp in existing:
            df_cp = pd.read_parquet(cp)
            all_records.extend(df_cp.to_dict("records"))
        skip = len(all_records)
        print(f"  Resuming from checkpoint: {skip:,} records already downloaded")

    batch_size = 1000
    checkpoint_interval = 100_000

    # Get total
    try:
        total = requests.get(f"{endpoint}?$count=true&$top=1", timeout=30).json()["metadata"]["count"]
        print(f"  Total NFIP records: {total:,}")
    except Exception:
        total = None

    while True:
        url = f"{endpoint}?$top={batch_size}&$skip={skip}&$format=json"
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n  FAILED at skip={skip}: {e}")
                    print("  Saving progress and exiting...")
                    _save_nfip_checkpoint(all_records, checkpoint_dir)
                    return
                time.sleep(10 * (attempt + 1))

        batch = resp.json().get(data_key, [])
        if not batch:
            break

        all_records.extend(batch)
        skip += len(batch)

        pct = f" ({skip/total*100:.1f}%)" if total else ""
        print(f"  Fetched {skip:,} records{pct}", end="\r")

        # Checkpoint every 100k rows
        if len(all_records) % checkpoint_interval < batch_size:
            _save_nfip_checkpoint(all_records, checkpoint_dir)

        if len(batch) < batch_size:
            break

        time.sleep(0.4)

    print(f"\n  Download complete: {len(all_records):,} NFIP claims")
    df = pd.DataFrame(all_records)
    df.to_parquet(out, index=False)
    print(f"  Saved → {out.relative_to(ROOT)}")

    # Clean up checkpoints
    for cp in glob.glob(str(checkpoint_dir / "checkpoint_*.parquet")):
        os.remove(cp)


def _save_nfip_checkpoint(records: list[dict], checkpoint_dir: Path) -> None:
    n = len(records)
    cp_path = checkpoint_dir / f"checkpoint_{n:08d}.parquet"
    pd.DataFrame(records).to_parquet(cp_path, index=False)
    print(f"\n  [checkpoint] Saved {n:,} records → {cp_path.name}")


# ---------------------------------------------------------------------------
# Datasets 6+7: Public Assistance Applicants and Projects
# ---------------------------------------------------------------------------

def download_public_assistance():
    datasets = {
        "pa_applicants": (
            "https://www.fema.gov/api/open/v1/PublicAssistanceApplicants",
            "PublicAssistanceApplicants",
        ),
        "pa_projects": (
            "https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsSummaries",
            "PublicAssistanceFundedProjectsSummaries",
        ),
    }
    for name, (endpoint, data_key) in datasets.items():
        out = RAW_FEMA / f"{name}.parquet"
        if out.exists():
            print(f"  {name}.parquet already exists — skipping")
            continue
        print(f"Downloading {name}...")
        records = openfema_paginate(endpoint, data_key=data_key)
        save_parquet(records, out, name)


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n--- Download Summary ---")
    expected = [
        "disaster_declarations.parquet",
        "housing_assistance_owners.parquet",
        "housing_assistance_renters.parquet",
        "ia_registrants_large_disasters.parquet",
        "nfip_claims.parquet",
        "pa_applicants.parquet",
        "pa_projects.parquet",
    ]
    all_ok = True
    for fname in expected:
        path = RAW_FEMA / fname
        if path.exists():
            n = len(pd.read_parquet(path, columns=["id"] if "nfip" not in fname else ["reportedZipCode"]).index)
            print(f"  ✓  {fname}: {n:,} rows")
        else:
            print(f"  ✗  {fname}: MISSING")
            all_ok = False

    if all_ok:
        print("\nAll FEMA datasets present.")
    else:
        print("\nSome datasets are missing — re-run to complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_disaster_declarations()
    download_housing_assistance()
    download_ia_registrants()
    download_public_assistance()
    download_nfip_claims()   # last — largest
    print_summary()

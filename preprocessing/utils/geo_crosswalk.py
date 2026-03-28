"""
geo_crosswalk.py
================
Build area-proportional geographic crosswalk tables for California research
projects that work at Census tract level.

All crosswalks follow the same convention:
  - Source unit column  (e.g., zcta5, county_fips, pctkey)
  - Target unit column  (tract_geoid_20 — 2020 Census tract GEOID)
  - weight column       (fraction of source unit area that falls in each target tract)
  - Weights per source unit sum to 1.0 (re-normalised after area computation)

Functions
---------
check_weights(df, unit_col, weight_col, label)
    Print weight-sum diagnostics. Useful to verify crosswalk integrity.

download_tiger_tracts(dest_dir, state_fips="06")
    Download 2020 TIGER/Line Census tract shapefile for a state.
    Returns path to the extracted directory.

build_zip_tract(output_path, shapes_dir)
    ZCTA → 2020 Census tract crosswalk using Census relationship file
    (area-proportional weights, re-normalised per ZCTA).

build_county_tract(acs_tracts_csv, output_path)
    County FIPS → 2020 Census tract lookup derived from an ACS tracts CSV.
    (No spatial join needed — county FIPS is the first 5 digits of the tract GEOID.)

build_prec_tract(vintage, prec_url, tiger_dir, output_path)
    Precinct → 2020 Census tract crosswalk via GeoPandas spatial overlay.
    Both layers projected to EPSG:3310 (CA Albers) before overlay.

Dependencies
------------
    geopandas  (pip install geopandas)
    pandas     (pip install pandas)
    requests   (pip install requests)
    download_utils  (from this utilities package)

Census data sources
-------------------
- ZCTA-to-tract relationship file (2020):
    https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt
- TIGER 2020 tract shapefiles:
    https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{state_fips}_tract.zip

Usage example
-------------
    from pathlib import Path
    from geo_crosswalk import build_zip_tract, build_county_tract

    SHAPES = Path("data/raw/shapefiles")
    PROCESSED = Path("data/processed")

    build_zip_tract(output_path=PROCESSED / "crosswalk_zip_tract.csv", shapes_dir=SHAPES)
    build_county_tract(
        acs_tracts_csv=Path("data/raw/acs/acs_tracts_ca_2020.csv"),
        output_path=PROCESSED / "crosswalk_county_tract.csv",
    )
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# CA Albers Equal Area — accurate area computation for California
CRS_CA = "EPSG:3310"

ZCTA_REL_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "zcta520/tab20_zcta520_tract20_natl.txt"
)
TIGER_TRACT_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{state_fips}_tract.zip"
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def check_weights(
    df: pd.DataFrame,
    unit_col: str,
    weight_col: str,
    label: str = "",
) -> None:
    """
    Print weight-sum diagnostics per source unit.

    Weights should sum to approximately 1.0 for each source unit after
    re-normalisation. Use this to verify crosswalk integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Crosswalk DataFrame.
    unit_col : str
        Column identifying the source geographic unit (e.g., "zcta5").
    weight_col : str
        Column containing area-proportional weights.
    label : str
        Optional label printed in the diagnostic output.
    """
    sums = df.groupby(unit_col)[weight_col].sum()
    tag = f" [{label}]" if label else ""
    print(f"  Weight-sum diagnostics{tag} — should be ≈1.0 per {unit_col}:")
    print(
        f"    n_units={len(sums):,}  "
        f"mean={sums.mean():.4f}  "
        f"min={sums.min():.4f}  "
        f"max={sums.max():.4f}  "
        f"pct_within_1pct={(abs(sums - 1.0) < 0.01).mean() * 100:.1f}%"
    )


# ---------------------------------------------------------------------------
# TIGER tract download
# ---------------------------------------------------------------------------


def download_tiger_tracts(dest_dir: Path, state_fips: str = "06") -> Path:
    """
    Download the 2020 TIGER/Line Census tract shapefile for *state_fips*.

    Parameters
    ----------
    dest_dir : Path
        Parent directory for shapefile extraction.
    state_fips : str
        Two-digit state FIPS code. Default "06" (California).

    Returns
    -------
    Path
        Path to the extraction directory containing the .shp file.
    """
    from download_utils import download_zip

    url = TIGER_TRACT_URL_TEMPLATE.format(state_fips=state_fips)
    name = f"tl_2020_{state_fips}_tract"
    return download_zip(url=url, dest_dir=dest_dir, name=name)


# ---------------------------------------------------------------------------
# Crosswalk 1: ZCTA → Census tract
# ---------------------------------------------------------------------------


def build_zip_tract(
    output_path: Path,
    shapes_dir: Path,
    state_fips: str = "06",
) -> pd.DataFrame:
    """
    Build a ZCTA (ZIP Code Tabulation Area) → 2020 Census tract crosswalk
    using the Census Bureau's official relationship file.

    Weights are area-proportional (AREALAND_PART / AREALAND_ZCTA), then
    re-normalised per ZCTA to sum exactly to 1.0.

    Parameters
    ----------
    output_path : Path
        Where to save the output CSV.
    shapes_dir : Path
        Directory to cache the downloaded relationship file.
    state_fips : str
        Two-digit state FIPS — used to filter the national relationship file.

    Returns
    -------
    pd.DataFrame
        Columns: zcta5, tract_geoid_20, weight.

    Output CSV columns
    ------------------
    zcta5           ZIP Code Tabulation Area (5-digit string)
    tract_geoid_20  11-digit Census tract GEOID (2020 definitions)
    weight          Fraction of ZCTA land area that falls in this tract
    """
    from download_utils import download_file

    output_path = Path(output_path)
    shapes_dir = Path(shapes_dir)

    print("\n=== Crosswalk: ZCTA → Census tract ===")

    # Download relationship file
    txt_file = shapes_dir / "tab20_zcta520_tract20_natl.txt"
    download_file(url=ZCTA_REL_URL, dest_path=txt_file)

    df = pd.read_csv(txt_file, sep="|", dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f"  Raw rows: {len(df):,}  columns: {list(df.columns)}")

    # Map columns (handle minor naming variants across Census vintages)
    col_map: dict = {}
    for col in df.columns:
        if "GEOID" in col and "ZCTA5" in col and "20" in col:
            col_map.setdefault("zcta", col)
        if "TRACT" in col and "20" in col and "GEOID" in col:
            col_map.setdefault("tract", col)
        if col == "AREALAND_PART":
            col_map["area_part"] = col
        if "AREALAND" in col and "ZCTA" in col and "PART" not in col:
            col_map.setdefault("area_zcta", col)

    required = ["zcta", "tract", "area_part", "area_zcta"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise KeyError(f"Could not identify columns for: {missing}. Available: {list(df.columns)}")

    df = df.rename(columns={
        col_map["zcta"]: "zcta5",
        col_map["tract"]: "tract_geoid_20",
        col_map["area_part"]: "area_part",
        col_map["area_zcta"]: "area_zcta",
    })

    # Filter to target state
    df = df[df["tract_geoid_20"].str.startswith(state_fips)].copy()
    print(f"  Rows after state filter ({state_fips}): {len(df):,}")

    # Compute weights
    df["area_part"] = pd.to_numeric(df["area_part"], errors="coerce").fillna(0)
    df["area_zcta"] = pd.to_numeric(df["area_zcta"], errors="coerce")
    df["weight"] = df["area_part"] / df["area_zcta"]

    # Re-normalise per ZCTA
    weight_sum = df.groupby("zcta5")["weight"].transform("sum")
    df.loc[weight_sum > 0, "weight"] = df.loc[weight_sum > 0, "weight"] / weight_sum[weight_sum > 0]
    df = df.drop(columns=["area_part", "area_zcta"])
    df = df[df["weight"] > 0].reset_index(drop=True)

    out = df[["zcta5", "tract_geoid_20", "weight"]].copy()
    check_weights(out, "zcta5", "weight", "ZIP→tract")
    print(f"  Output: {len(out):,} rows, {out['zcta5'].nunique():,} unique ZCTAs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")
    return out


# ---------------------------------------------------------------------------
# Crosswalk 2: County FIPS → Census tract
# ---------------------------------------------------------------------------


def build_county_tract(
    acs_tracts_csv: Path,
    output_path: Path,
    state_fips: str = "06",
) -> pd.DataFrame:
    """
    Build a county FIPS → 2020 Census tract lookup from an ACS tracts CSV.

    No spatial join required: county FIPS is simply the first 5 digits of
    each tract's 11-digit GEOID. Use this crosswalk to broadcast county-level
    data (e.g., insurance panels) to all tracts within each county.

    Parameters
    ----------
    acs_tracts_csv : Path
        Path to a CSV that contains one row per Census tract with a GEOID
        column (11-digit string, starting with the state FIPS code).
        Typically produced by census_api.fetch_acs_tracts().
    output_path : Path
        Where to save the output CSV.
    state_fips : str
        Two-digit state FIPS to filter the tract list. Default "06" (CA).

    Returns
    -------
    pd.DataFrame
        Columns: county_fips (5-digit str), tract_geoid_20 (11-digit str).

    Output CSV columns
    ------------------
    county_fips     5-digit county FIPS (state 2 + county 3)
    tract_geoid_20  11-digit Census tract GEOID
    """
    print("\n=== Crosswalk: County FIPS → Census tract ===")

    df = pd.read_csv(acs_tracts_csv, dtype=str, low_memory=False)
    print(f"  ACS file rows: {len(df):,}")

    # Find GEOID column
    geoid_col = None
    for c in df.columns:
        if c.upper() in ("GEOID", "GEO_ID", "TRACT_GEOID", "FIPS"):
            geoid_col = c
            break
    if geoid_col is None:
        for c in df.columns:
            sample = df[c].dropna().iloc[0] if len(df) > 0 else ""
            if str(sample).startswith(state_fips) and len(str(sample)) >= 11:
                geoid_col = c
                print(f"  Inferred GEOID column: '{c}'  sample={sample!r}")
                break
    if geoid_col is None:
        raise KeyError(f"Could not identify GEOID column. Columns: {list(df.columns)}")

    geoids = df[geoid_col].dropna().astype(str).str.strip()
    # Strip "1400000US" prefix if present (Census API GeoID format)
    geoids = geoids.str.replace(r"^1400000US", "", regex=True)
    ca_geoids = geoids[geoids.str.startswith(state_fips)]

    out = pd.DataFrame({
        "county_fips": ca_geoids.str[:5].values,
        "tract_geoid_20": ca_geoids.values,
    }).drop_duplicates().reset_index(drop=True)

    print(
        f"  Unique county FIPS: {out['county_fips'].nunique()} "
        f"(expect 58 for CA)  |  tracts: {len(out):,}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")
    return out


# ---------------------------------------------------------------------------
# Crosswalk 3: Precinct → Census tract (spatial overlay)
# ---------------------------------------------------------------------------


def build_prec_tract(
    vintage: str,
    prec_url: str,
    tiger_dir: Path,
    output_path: Path,
    state_fips: str = "06",
) -> pd.DataFrame:
    """
    Build a precinct → 2020 Census tract crosswalk via GeoPandas spatial overlay.

    Both shapefiles are projected to EPSG:3310 (CA Albers Equal Area) before
    overlay for accurate area computation.

    Weight = intersection area / precinct total area, re-normalised per precinct.

    Parameters
    ----------
    vintage : str
        Short label for the precinct vintage (e.g., "g22", "p18"). Used in
        output filenames and log messages.
    prec_url : str
        URL to download the precinct shapefile ZIP.
        Example (UC Berkeley Statewide Database):
          g22: https://statewidedatabase.org/pub/data/G22/state/mprec_state_g22_v01_shp.zip
          p18: https://statewidedatabase.org/pub/data/P18/state/mprec_p18_v01_shp.zip
    tiger_dir : Path
        Directory containing (or where to download) the TIGER tract shapefile.
        The function calls download_tiger_tracts() automatically if the shapefile
        is not already present.
    output_path : Path
        Where to save the output CSV.
    state_fips : str
        Two-digit state FIPS code. Default "06" (California).

    Returns
    -------
    pd.DataFrame
        Columns: pctkey, tract_geoid_20, weight.

    Output CSV columns
    ------------------
    pctkey          Precinct identifier (string, from shapefile key column)
    tract_geoid_20  11-digit Census tract GEOID (2020 definitions)
    weight          Fraction of precinct area that falls in this tract
    """
    import geopandas as gpd
    from download_utils import download_zip

    print(f"\n=== Crosswalk: Precinct → Census tract ({vintage}) ===")

    prec_dir = download_zip(url=prec_url, dest_dir=tiger_dir, name=f"prec_{vintage}")
    tract_dir = download_tiger_tracts(dest_dir=tiger_dir, state_fips=state_fips)

    # Load tracts
    shp_files = list(tract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp in {tract_dir}")
    tracts = gpd.read_file(shp_files[0])
    geoid_col = next((c for c in tracts.columns if c.upper() == "GEOID"), None)
    if geoid_col is None:
        raise KeyError(f"No GEOID column in tracts shapefile. Columns: {list(tracts.columns)}")
    tracts = tracts[[geoid_col, "geometry"]].rename(columns={geoid_col: "tract_geoid_20"})
    print(f"  Tracts: {len(tracts):,}  CRS: {tracts.crs}")

    # Load precincts
    prec_shp = list(prec_dir.rglob("*.shp"))
    if not prec_shp:
        raise FileNotFoundError(f"No .shp in {prec_dir}")
    precincts = gpd.read_file(prec_shp[0])
    print(f"  Precincts: {len(precincts):,}  CRS: {precincts.crs}")

    # Identify precinct key column
    key_candidates = ["PCTKEY", "PCTFIPS", "PCT_KEY", "PREC_KEY", "MPREC_KEY", "PRECINCT_ID", "PRECKEY"]
    pct_key_col = None
    for c in precincts.columns:
        if c.upper() in [k.upper() for k in key_candidates]:
            pct_key_col = c
            break
    if pct_key_col is None:
        for c in precincts.columns:
            if c != "geometry" and precincts[c].dtype == object:
                pct_key_col = c
                print(f"  WARNING: No standard key column found; using '{c}' as pctkey")
                break
    print(f"  Precinct key column: '{pct_key_col}'")

    precincts = precincts[[pct_key_col, "geometry"]].rename(columns={pct_key_col: "pctkey"})

    # Project to CA Albers
    tracts = tracts.to_crs(CRS_CA)
    precincts = precincts.to_crs(CRS_CA)

    prec_areas = (
        precincts[["pctkey"]]
        .copy()
        .assign(prec_area_total=precincts.geometry.area)
        .drop_duplicates("pctkey")
    )

    print(f"  Running spatial overlay ({len(precincts):,} precincts × {len(tracts):,} tracts)...")
    overlay = gpd.overlay(
        precincts[["pctkey", "geometry"]],
        tracts,
        how="intersection",
        keep_geom_type=False,
    )
    overlay["area_part"] = overlay.geometry.area
    overlay = overlay.merge(prec_areas, on="pctkey", how="left")
    overlay["weight"] = overlay["area_part"] / overlay["prec_area_total"]

    weight_sum = overlay.groupby("pctkey")["weight"].transform("sum")
    overlay.loc[weight_sum > 0, "weight"] = (
        overlay.loc[weight_sum > 0, "weight"] / weight_sum[weight_sum > 0]
    )
    overlay = overlay[overlay["weight"] > 1e-5].copy()

    out = overlay[["pctkey", "tract_geoid_20", "weight"]].reset_index(drop=True)
    check_weights(out, "pctkey", "weight", f"precinct({vintage})→tract")
    print(f"  Output: {len(out):,} rows, {out['pctkey'].nunique():,} unique precincts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")
    return out

"""
arcgis_rest.py
==============
Helpers for querying ArcGIS REST feature services — specifically for paginated
GeoJSON downloads from ArcGIS MapServer and FeatureServer endpoints.

Many California state agency datasets (CalFire FHSZ, CalEnviroScreen, DWR,
CDFW) are served via ArcGIS REST. This module handles pagination transparently
so callers receive a complete FeatureCollection regardless of server page limits.

Functions
---------
get_record_count(base_url)
    Return the total number of features in an ArcGIS layer.

paginate_geojson(base_url, out_fields, page_size, sleep)
    Fetch all features from an ArcGIS REST endpoint with automatic pagination.
    Returns a GeoJSON FeatureCollection dict.

save_geojson(feature_collection, dest_path)
    Write a GeoJSON FeatureCollection to a file, skipping if already present.

Dependencies
------------
    Standard library: json, time, urllib.request, pathlib

No third-party dependencies — uses only urllib so it works in minimal
environments without requests installed.

Usage example
-------------
    from arcgis_rest import paginate_geojson, save_geojson
    from pathlib import Path

    # CalFire Fire Hazard Severity Zones (SRA, 2007 designations)
    SRA_URL = (
        "https://services.gis.ca.gov/arcgis/rest/services/Environment/"
        "Fire_Severity_Zones/MapServer/0/query"
    )

    gj = paginate_geojson(
        base_url=SRA_URL,
        out_fields="HAZ_CLASS,SRA",
        page_size=1000,
    )
    save_geojson(gj, Path("data/raw/calfire_fhsz/fhsz_sra.geojson"))

    # Load into GeoPandas
    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

ArcGIS REST API notes
---------------------
- The query endpoint is always: {service_url}/{layer_id}/query
- Parameters:
    where=1=1           — return all features (no filter)
    outFields=*         — return all fields (or comma-separated list)
    f=geojson           — GeoJSON output format
    returnGeometry=true — include geometry
    resultOffset=N      — pagination offset
    resultRecordCount=N — page size (server may cap this; 1000 is usually safe)
- To find a service URL: browse the ArcGIS REST directory at
    {server}/arcgis/rest/services
  or use the ArcGIS Online item page "View" → "Service URL"
- Some services require authentication. This module does not handle OAuth/token
  auth — for authenticated services, use the ArcGIS Python API instead.
"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Optional


def get_record_count(base_url: str) -> int:
    """
    Return the total feature count for an ArcGIS REST layer.

    Parameters
    ----------
    base_url : str
        Base query URL, e.g.
        "https://services.gis.ca.gov/arcgis/rest/services/Environment/
         Fire_Severity_Zones/MapServer/0/query"

    Returns
    -------
    int
        Total number of features reported by the server.
    """
    url = f"{base_url}?where=1%3D1&returnCountOnly=true&f=json"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    count = data.get("count", 0)
    return count


def _build_page_url(base_url: str, offset: int, count: int, out_fields: str) -> str:
    params = (
        f"where=1%3D1"
        f"&outFields={out_fields}"
        f"&f=geojson"
        f"&returnGeometry=true"
        f"&resultOffset={offset}"
        f"&resultRecordCount={count}"
    )
    return f"{base_url}?{params}"


def paginate_geojson(
    base_url: str,
    out_fields: str = "*",
    page_size: int = 1000,
    sleep: float = 0.2,
) -> dict:
    """
    Fetch all features from an ArcGIS REST endpoint with automatic pagination.

    Queries the layer in pages of *page_size* features, concatenates all
    features, and returns a complete GeoJSON FeatureCollection dict.

    Parameters
    ----------
    base_url : str
        ArcGIS REST query endpoint URL (ending in /query).
    out_fields : str
        Comma-separated field names to return, or "*" for all fields.
        Example: "HAZ_CLASS,SRA"
    page_size : int
        Number of features to request per page. Default 1000.
        Most ArcGIS servers cap this at 1000–2000 regardless of what you request.
    sleep : float
        Seconds to sleep between page requests. Default 0.2.
        Increase if you encounter rate-limit errors from a public service.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with keys: "type", "features", "crs".
        "crs" is taken from the last page response (usually EPSG:4326).

    Raises
    ------
    urllib.error.HTTPError
        If the server returns an HTTP error status.
    json.JSONDecodeError
        If the server returns malformed JSON (e.g., an error message).

    Notes
    -----
    - If the server returns fewer features than expected (e.g., due to a
      server-side record limit), the function stops early rather than looping
      infinitely.
    - Progress is printed to stdout as features are downloaded.
    """
    total = get_record_count(base_url)
    print(f"  Total features: {total:,}")

    all_features = []
    offset = 0
    last_chunk = {}

    while offset < total:
        url = _build_page_url(base_url, offset, page_size, out_fields)
        with urllib.request.urlopen(url) as resp:
            last_chunk = json.loads(resp.read())

        features = last_chunk.get("features", [])
        if not features:
            print(f"\n  WARNING: Server returned 0 features at offset {offset}; stopping.")
            break

        all_features.extend(features)
        offset += len(features)
        print(f"  Downloaded {offset:,}/{total:,} features", end="\r")

        if sleep > 0:
            time.sleep(sleep)

    print()  # newline after progress line
    return {
        "type": "FeatureCollection",
        "features": all_features,
        "crs": last_chunk.get(
            "crs", {"type": "name", "properties": {"name": "EPSG:4326"}}
        ),
    }


def save_geojson(
    feature_collection: dict,
    dest_path: Path,
    skip_if_exists: bool = True,
) -> Path:
    """
    Write a GeoJSON FeatureCollection dict to *dest_path*.

    Parameters
    ----------
    feature_collection : dict
        A GeoJSON FeatureCollection as returned by paginate_geojson().
    dest_path : Path
        Destination file path (.geojson or .json).
    skip_if_exists : bool
        If True (default), skip writing if the file already exists.

    Returns
    -------
    Path
        *dest_path*.
    """
    dest_path = Path(dest_path)
    if skip_if_exists and dest_path.exists():
        print(f"  [skip] {dest_path.name} already present")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(feature_collection, f)

    n = len(feature_collection.get("features", []))
    print(f"  [saved] {dest_path.name} ({n:,} features)")
    return dest_path


def load_geojson(path: Path) -> dict:
    """
    Load a GeoJSON file from disk.

    Parameters
    ----------
    path : Path
        Path to a .geojson or .json file.

    Returns
    -------
    dict
        Parsed GeoJSON dict.
    """
    with open(path) as f:
        return json.load(f)

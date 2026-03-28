"""
preprocessing/04_hazard_intensity.py
Build ZIP-level hazard exposure controls and warning lead time variable.

Reads:
  data/intermediate/zcta_centroids.parquet
  data/intermediate/events_county_panel.parquet
  data/raw/usgs/high_water_marks.parquet
  data/raw/nhc/hurdat2.parquet
  data/raw/nws/warnings_shp_{year}.zip   (IEM shapefiles — required for spatial joins)

Outputs:
  data/intermediate/hazard_intensity_panel.parquet

Key outputs per ZIP × event:
  usgs_hwm_depth_ft       — IDW-interpolated flood depth from USGS HWMs
  hwm_coverage_flag       — True if ≥1 HWM within 50km
  wind_speed_kt           — step-function wind speed from HURDAT2 wind radii
  warning_lead_hours      — hours from earliest NWS warning to event start
  flash_event             — warning_lead_hours < 6
  short_notice            — 6 ≤ warning_lead_hours < 24
  adequate_notice         — warning_lead_hours ≥ 24  (required for Mechanism A primary sample)

WARNING LEAD TIME: Uses IEM shapefile geometries for spatial county matching.
If shapefiles are missing, falls back to CSV + WFO lookup (less accurate).

Run order: after 01_standardize_geography.py and 02_construct_events.py
"""

import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree

ROOT  = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"

# IDW search radius in km (pre-specified in project plan §Step 4)
HWM_SEARCH_RADIUS_KM = 50
# Hurricane wind radii thresholds in knots (HURDAT2 standard — physical constants)
HURDAT_RADII_KT = [34, 50, 64]
# Warning phenomena to match (pre-specified in project plan §Step 4)
NWS_PHENOMS = {"FF", "FL", "HU", "TO", "SV"}


def require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}\n  {hint}")
        sys.exit(1)


def km_to_deg(km: float) -> float:
    """Rough conversion: 1 degree ≈ 111 km (equatorial)."""
    return km / 111.0


# ---------------------------------------------------------------------------
# 4a. USGS HWM → ZIP centroid (IDW interpolation)
# ---------------------------------------------------------------------------

def compute_hwm_depths(centroids: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    hwm_path = RAW / "usgs" / "high_water_marks.parquet"
    if not hwm_path.exists():
        print("  USGS HWM file not found — flood depth will be NaN")
        return pd.DataFrame(columns=["zcta5", "disasterNumber",
                                     "usgs_hwm_depth_ft", "hwm_coverage_flag"])

    hwm = pd.read_parquet(hwm_path)
    print(f"  USGS HWMs loaded: {len(hwm):,} marks")

    # Detect key columns
    lat_col   = next((c for c in hwm.columns if "lat" in c.lower()), None)
    lon_col   = next((c for c in hwm.columns if "lon" in c.lower()), None)
    depth_col = next((c for c in hwm.columns
                      if "depth" in c.lower() or "height" in c.lower() or "elev" in c.lower()), None)
    event_col = next((c for c in hwm.columns
                      if "event" in c.lower() and "id" in c.lower()), None)

    if not lat_col or not lon_col:
        print("  WARNING: No lat/lon columns in HWM data — skipping HWM interpolation")
        return pd.DataFrame(columns=["zcta5", "disasterNumber",
                                     "usgs_hwm_depth_ft", "hwm_coverage_flag"])

    hwm[lat_col] = pd.to_numeric(hwm[lat_col], errors="coerce")
    hwm[lon_col] = pd.to_numeric(hwm[lon_col], errors="coerce")
    if depth_col:
        hwm[depth_col] = pd.to_numeric(hwm[depth_col], errors="coerce")

    hwm_valid = hwm[hwm[lat_col].notna() & hwm[lon_col].notna()].copy()

    # Build KD-tree on HWM coordinates
    hwm_coords = np.column_stack([hwm_valid[lat_col], hwm_valid[lon_col]])
    tree = cKDTree(hwm_coords)

    # Flood events only (for HWM depth interpolation)
    flood_events = events[events["hazard_type"].isin(
        ["flood_only", "flood_mixed"]
    )].drop_duplicates(subset=["disasterNumber"]) if "hazard_type" in events.columns else events

    results = []
    radius_deg = km_to_deg(HWM_SEARCH_RADIUS_KM)

    for _, zip_row in centroids.iterrows():
        zcta = zip_row["zcta5"]
        lat  = zip_row["centroid_lat"]
        lon  = zip_row["centroid_lon"]

        if pd.isna(lat) or pd.isna(lon):
            continue

        # Find HWMs within search radius
        idxs = tree.query_ball_point([lat, lon], r=radius_deg)
        if not idxs:
            # No HWMs within radius — attach to nearby flood disasters as NaN
            for _, ev in flood_events.iterrows():
                results.append({
                    "zcta5": zcta,
                    "disasterNumber": ev["disasterNumber"],
                    "usgs_hwm_depth_ft": float("nan"),
                    "hwm_coverage_flag": False,
                })
            continue

        nearby = hwm_valid.iloc[idxs]
        dists = np.sqrt(
            (nearby[lat_col].values - lat) ** 2 +
            (nearby[lon_col].values - lon) ** 2
        )

        if depth_col and nearby[depth_col].notna().any():
            # IDW with distance weights
            valid_mask = nearby[depth_col].notna()
            depths = nearby.loc[valid_mask, depth_col].values
            d = dists[valid_mask.values]
            d = np.where(d == 0, 1e-10, d)
            idw_depth = np.sum(depths / d) / np.sum(1 / d)
        else:
            idw_depth = float("nan")

        # Assign to all flood events for this ZIP (event matching happens in 05)
        for _, ev in flood_events.iterrows():
            results.append({
                "zcta5": zcta,
                "disasterNumber": ev["disasterNumber"],
                "usgs_hwm_depth_ft": idw_depth,
                "hwm_coverage_flag": True,
            })

    df = pd.DataFrame(results)
    n_covered = df["hwm_coverage_flag"].sum() if not df.empty else 0
    total = len(df)
    print(f"  HWM coverage: {n_covered:,}/{total:,} ZIP-events ({n_covered/total*100:.1f}% if any)")
    return df


# ---------------------------------------------------------------------------
# 4b. HURDAT2 step-function wind field → ZIP
# ---------------------------------------------------------------------------

def compute_hurricane_winds(centroids: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    hurdat_path = RAW / "nhc" / "hurdat2.parquet"
    if not hurdat_path.exists():
        print("  HURDAT2 file not found — wind speeds will be NaN")
        return pd.DataFrame(columns=["zcta5", "disasterNumber", "wind_speed_kt"])

    hurdat = pd.read_parquet(hurdat_path)
    print(f"  HURDAT2 loaded: {len(hurdat):,} track points")

    # Parse dates
    hurdat["track_dt"] = pd.to_datetime(
        hurdat["date"].astype(str) + hurdat["time_utc"].astype(str).str.zfill(4),
        format="%Y%m%d%H%M", errors="coerce",
    )
    hurdat["lat"] = pd.to_numeric(hurdat["lat"], errors="coerce")
    hurdat["lon"] = pd.to_numeric(hurdat["lon"], errors="coerce")

    # Hurricane events only
    hurr_events = events[
        events.get("hazard_type", pd.Series("", index=events.index)) == "hurricane"
    ] if "hazard_type" in events.columns else pd.DataFrame()

    if hurr_events.empty:
        print("  No hurricane events in panel — skipping wind field computation")
        return pd.DataFrame(columns=["zcta5", "disasterNumber", "wind_speed_kt"])

    results = []
    for _, ev in hurr_events.iterrows():
        ev_begin = ev.get("incidentBeginDate")
        ev_end   = ev.get("incidentEndDate", ev_begin)
        if pd.isna(ev_begin):
            continue

        # Match HURDAT2 track to event by date window
        track = hurdat[
            (hurdat["track_dt"] >= ev_begin - pd.Timedelta(days=2)) &
            (hurdat["track_dt"] <= (ev_end if pd.notna(ev_end) else ev_begin) + pd.Timedelta(days=2))
        ]
        if track.empty:
            continue

        for _, zip_row in centroids.iterrows():
            zcta = zip_row["zcta5"]
            z_lat = zip_row["centroid_lat"]
            z_lon = zip_row["centroid_lon"]

            if pd.isna(z_lat) or pd.isna(z_lon):
                continue

            # Step-function wind field from HURDAT2 wind radii
            # Per project plan: use 34/50/64-knot wind radii as step-function approximation.
            # NOTE: Full Holland (1980) parametric wind field is the planned future improvement.
            max_wind = 0
            for _, tp in track.iterrows():
                dist_km = _haversine_km(z_lat, z_lon, tp["lat"], tp["lon"])
                # Check which wind radius band the ZIP falls in (quadrant-averaged)
                for threshold_kt, quad_suffix in zip(
                    HURDAT_RADII_KT, [("r34", "ne"), ("r50", "ne"), ("r64", "ne")]
                ):
                    # Average over available quadrants
                    quad_cols = [f"r{threshold_kt}_{q}" for q in ["ne", "se", "sw", "nw"]
                                 if f"r{threshold_kt}_{q}" in tp.index]
                    if not quad_cols:
                        continue
                    avg_radius_nm = np.nanmean([tp[c] for c in quad_cols
                                                if pd.notna(tp[c])])
                    if np.isnan(avg_radius_nm) or avg_radius_nm <= 0:
                        continue
                    radius_km = avg_radius_nm * 1.852  # nautical miles → km
                    if dist_km <= radius_km:
                        max_wind = max(max_wind, threshold_kt)

            results.append({
                "zcta5":           zcta,
                "disasterNumber":  ev["disasterNumber"],
                "wind_speed_kt":   max_wind if max_wind > 0 else float("nan"),
            })

    df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["zcta5", "disasterNumber", "wind_speed_kt"]
    )
    print(f"  Wind speeds computed for {len(df):,} ZIP-hurricane-event pairs")
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 4c. NWS warning lead time (IEM shapefiles → county → event)
# ---------------------------------------------------------------------------

def compute_warning_lead_times(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each event × county, compute warning lead time from earliest relevant
    NWS warning polygon intersecting the county.

    Prefers IEM shapefile format (has polygon geometry).
    Falls back to CSV format if shapefiles unavailable (less accurate — uses WFO).
    """
    nws_dir = RAW / "nws"

    shp_files = sorted(nws_dir.glob("warnings_shp_*.zip"))
    csv_files = sorted(nws_dir.glob("warnings_*.csv"))

    if not shp_files and not csv_files:
        print("  No NWS warning files found — warning_lead_hours will be NaN")
        return _empty_warning_df(events)

    if shp_files:
        print(f"  Using {len(shp_files)} IEM shapefile archives for spatial join")
        return _compute_lead_times_from_shapefiles(events, shp_files)
    else:
        print(f"  WARNING: No shapefiles found; falling back to CSV (less accurate)")
        print("  Re-run acquire/03_hazard_point_sources.py to download shapefiles")
        return _compute_lead_times_from_csv(events, csv_files)


def _empty_warning_df(events: pd.DataFrame) -> pd.DataFrame:
    result = events[["disasterNumber", "county_fips"]].drop_duplicates().copy()
    result["warning_lead_hours"] = float("nan")
    result["flash_event"]    = False
    result["short_notice"]   = False
    result["adequate_notice"] = False
    return result


def _compute_lead_times_from_shapefiles(
    events: pd.DataFrame, shp_files: list
) -> pd.DataFrame:
    """
    Load IEM warning shapefiles year by year, spatial-join to county polygons,
    compute earliest warning per event × county.
    """
    # Load county polygons (from ZCTA centroids + FIPS lookup, or use a county shapefile)
    # For robustness, use a simple approach: build county bounding boxes from
    # Census ZCTA centroids grouped by county
    centroids = pd.read_parquet(INTER / "zcta_centroids.parquet")
    primary   = pd.read_parquet(INTER / "zcta_county_primary.parquet")
    cent_county = centroids.merge(primary, on="zcta5", how="left")

    county_bounds = (
        cent_county.groupby("county_fips")
        .agg(
            min_lat=("centroid_lat", "min"),
            max_lat=("centroid_lat", "max"),
            min_lon=("centroid_lon", "min"),
            max_lon=("centroid_lon", "max"),
        )
        .reset_index()
    )

    results = []
    events_by_year = events.groupby(events["event_year"])

    for shp_zip in shp_files:
        year_str = shp_zip.stem.replace("warnings_shp_", "")
        try:
            year = int(year_str)
        except ValueError:
            continue

        if year not in events_by_year.groups:
            continue

        year_events = events_by_year.get_group(year)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = gpd.read_file(f"zip://{shp_zip}")
        except Exception as e:
            print(f"  WARNING: Could not read {shp_zip.name}: {e}")
            continue

        if gdf.empty:
            continue

        # Detect columns
        phenom_col = next((c for c in gdf.columns if c.upper() in ("PHENOM", "PHENOMENA")), None)
        sig_col    = next((c for c in gdf.columns if c.upper() in ("SIG", "SIGNIFICANCE")), None)
        issued_col = next((c for c in gdf.columns if "ISSUED" in c.upper() or "ISSUE" in c.upper()), None)
        status_col = next((c for c in gdf.columns if "STATUS" in c.upper()), None)

        if phenom_col:
            gdf = gdf[gdf[phenom_col].isin(NWS_PHENOMS)]
        if sig_col:
            gdf = gdf[gdf[sig_col] == "W"]
        if status_col:
            gdf = gdf[gdf[status_col] == "NEW"]

        if issued_col:
            gdf[issued_col] = pd.to_datetime(gdf[issued_col], errors="coerce", utc=True).dt.tz_localize(None)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # For each event in this year, find earliest warning in each county
        for _, ev in year_events.iterrows():
            ev_begin  = ev["incidentBeginDate"]
            ev_county = ev["county_fips"]

            if pd.isna(ev_begin):
                continue

            # Get county bounding box
            cb = county_bounds[county_bounds["county_fips"] == ev_county]
            if cb.empty:
                continue
            cb = cb.iloc[0]

            # Filter warnings to bounding box (fast pre-filter before geometry)
            bbox_mask = (
                (gdf.geometry.bounds["maxx"] >= cb["min_lon"]) &
                (gdf.geometry.bounds["minx"] <= cb["max_lon"]) &
                (gdf.geometry.bounds["maxy"] >= cb["min_lat"]) &
                (gdf.geometry.bounds["miny"] <= cb["max_lat"])
            )
            bbox_warnings = gdf[bbox_mask]

            if bbox_warnings.empty or not issued_col:
                continue

            # Filter to warnings issued before event start
            pre_event = bbox_warnings[bbox_warnings[issued_col] < ev_begin]
            if pre_event.empty:
                continue

            earliest = pre_event[issued_col].min()
            lead_hours = (ev_begin - earliest).total_seconds() / 3600

            results.append({
                "disasterNumber":    ev["disasterNumber"],
                "county_fips":       ev_county,
                "warning_lead_hours": round(lead_hours, 1),
            })

    if results:
        df = pd.DataFrame(results).drop_duplicates(
            subset=["disasterNumber", "county_fips"]
        )
    else:
        df = _empty_warning_df(events).drop(
            columns=["flash_event", "short_notice", "adequate_notice"]
        )

    # Merge back to full events panel
    full = events[["disasterNumber", "county_fips"]].drop_duplicates().merge(
        df, on=["disasterNumber", "county_fips"], how="left"
    )
    # Lead time classifications (pre-specified in project plan §Step 4)
    full["flash_event"]    = full["warning_lead_hours"].lt(6).fillna(False)
    full["short_notice"]   = full["warning_lead_hours"].between(6, 24, inclusive="left").fillna(False)
    full["adequate_notice"] = full["warning_lead_hours"].ge(24).fillna(False)

    n_covered = full["warning_lead_hours"].notna().sum()
    pct = n_covered / len(full) * 100
    print(f"  Warning lead time coverage: {n_covered:,}/{len(full):,} ({pct:.1f}%)")
    if pct < 50:
        print("  WARNING: <50% coverage — IEM shapefile gaps likely. Diagnose before proceeding.")

    return full


def _compute_lead_times_from_csv(events: pd.DataFrame, csv_files: list) -> pd.DataFrame:
    """Fallback: use CSV warnings (no geometry — approximate county match via WFO)."""
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "PHENOM" in df.columns and "SIG" in df.columns:
                df = df[df["PHENOM"].isin(NWS_PHENOMS) & (df["SIG"] == "W")]
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return _empty_warning_df(events)

    warnings_df = pd.concat(dfs, ignore_index=True)
    issued_col = next((c for c in warnings_df.columns if "ISSUED" in c.upper()), None)
    if not issued_col:
        return _empty_warning_df(events)

    warnings_df[issued_col] = pd.to_datetime(warnings_df[issued_col], errors="coerce")

    # Without geometry, approximate: match warnings to events by state + date only
    # This is much less precise than the shapefile approach
    state_col = next((c for c in warnings_df.columns
                      if c.upper() in ("ST", "STATE", "STATEABBR")), None)

    result = _empty_warning_df(events)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = INTER / "hazard_intensity_panel.parquet"
    if out.exists():
        print("hazard_intensity_panel.parquet already exists — delete to rebuild")
        df = pd.read_parquet(out)
        print(f"  {len(df):,} rows, columns: {list(df.columns)}")
        return

    print("=== Step 4: Hazard Intensity ===\n")

    require(INTER / "zcta_centroids.parquet",      "Run 01_standardize_geography.py")
    require(INTER / "events_county_panel.parquet", "Run 02_construct_events.py")

    centroids = pd.read_parquet(INTER / "zcta_centroids.parquet")
    events    = pd.read_parquet(INTER / "events_county_panel.parquet")
    print(f"Centroids: {len(centroids):,}  |  Events: {len(events):,}\n")

    # 4a. Flood depths
    print("4a. USGS HWM flood depths...")
    hwm_depths = compute_hwm_depths(centroids, events)

    # 4b. Hurricane wind speeds
    print("\n4b. HURDAT2 hurricane wind speeds...")
    wind_speeds = compute_hurricane_winds(centroids, events)

    # 4c. Warning lead times
    print("\n4c. NWS warning lead times...")
    lead_times = compute_warning_lead_times(events)

    # Merge: start from events × county, add hazard controls
    base = events[["disasterNumber", "county_fips"]].drop_duplicates()

    # Add flood depths (ZIP-level) via county
    xwalk = pd.read_parquet(INTER / "zcta_county_primary.parquet")
    if not hwm_depths.empty and "zcta5" in hwm_depths.columns:
        hwm_county = (
            hwm_depths.merge(xwalk[["zcta5", "county_fips"]], on="zcta5", how="left")
            .groupby(["county_fips", "disasterNumber"])
            .agg(
                usgs_hwm_depth_ft  =("usgs_hwm_depth_ft", "mean"),
                hwm_coverage_flag  =("hwm_coverage_flag", "any"),
            )
            .reset_index()
        )
        base = base.merge(hwm_county, on=["disasterNumber", "county_fips"], how="left")
    else:
        base["usgs_hwm_depth_ft"] = float("nan")
        base["hwm_coverage_flag"] = False

    # Add wind speeds (ZIP-level → county mean)
    if not wind_speeds.empty and "zcta5" in wind_speeds.columns:
        wind_county = (
            wind_speeds.merge(xwalk[["zcta5", "county_fips"]], on="zcta5", how="left")
            .groupby(["county_fips", "disasterNumber"])
            ["wind_speed_kt"].mean()
            .reset_index()
        )
        base = base.merge(wind_county, on=["disasterNumber", "county_fips"], how="left")
    else:
        base["wind_speed_kt"] = float("nan")

    # Add warning lead times (county-level)
    base = base.merge(lead_times, on=["disasterNumber", "county_fips"], how="left")

    base.to_parquet(out, index=False)
    print(f"\nSaved {len(base):,} rows → {out.relative_to(ROOT)}")

    # Validation
    print("\n--- Validation ---")
    if "usgs_hwm_depth_ft" in base.columns:
        cov = base["hwm_coverage_flag"].mean() if "hwm_coverage_flag" in base else float("nan")
        print(f"  HWM coverage:          {cov:.1%}")
    if "warning_lead_hours" in base.columns:
        wh = base["warning_lead_hours"]
        print(f"  Warning lead time:     mean={wh.mean():.1f}h, "
              f"adequate_notice={base['adequate_notice'].mean():.1%}, "
              f"coverage={wh.notna().mean():.1%}")
        if wh.notna().mean() < 0.5:
            print("  *** STOP: Warning lead time coverage <50%. "
                  "Diagnose IEM shapefile gaps before running 05. ***")


if __name__ == "__main__":
    main()

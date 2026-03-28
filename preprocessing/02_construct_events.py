"""
preprocessing/02_construct_events.py
Build the event-county panel — the spine of both analysis datasets.

Reads:
  data/raw/fema/disaster_declarations.parquet
  data/raw/fema/pa_projects.parquet
  data/raw/noaa/storm_events_details.parquet
  data/raw/census/acs_zcta_demographics.parquet  (population for per-capita damage)
  data/raw/elections/county_pres_returns.csv
  data/raw/elections/klarner_state_politics.csv
  data/intermediate/zcta_county_primary.parquet
  data/intermediate/fips_standardized.parquet

Outputs:
  data/intermediate/events_county_panel.parquet

Running variable: raw per_capita_damage (placeholder — TTR adjustment in 04b).
Historical IA linkage: self-join to find most recent prior event per county.
Temporal decay bins pre-specified in project plan §Research Design (Gallagher 2014).

Run order: after 01_standardize_geography.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"
INTER.mkdir(exist_ok=True)

# Gallagher (2014) temporal decay bins — pre-specified in project plan §Research Design
# (confirmed by PI in project plan: "Bin years_since_prior into: 0-2, 3-5, 6-9, 10+ years")
TEMPORAL_BINS  = [0, 2, 5, 9, 15, float("inf")]
TEMPORAL_LABELS = ["0-2yr", "3-5yr", "6-9yr", "10-15yr", "15+yr"]

# Event-to-NOAA match window: ±3 days (pre-specified in project plan §Step 2)
NOAA_MATCH_DAYS = 3


def require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}\n  {hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2a. Reshape FEMA declarations → one row per disaster × county
# ---------------------------------------------------------------------------

def load_fema_declarations() -> pd.DataFrame:
    path = RAW / "fema" / "disaster_declarations.parquet"
    require(path, "Run acquire/01_fema.py")

    df = pd.read_parquet(path)
    print(f"  Raw declarations: {len(df):,} rows")

    # Restrict to DR (major disaster) declarations
    if "declarationType" in df.columns:
        df = df[df["declarationType"] == "DR"].copy()
        print(f"  After DR filter: {len(df):,} rows")

    # Construct 5-digit county FIPS
    df["county_fips"] = (
        df["fipsStateCode"].astype(str).str.zfill(2) +
        df["fipsCountyCode"].astype(str).str.zfill(3)
    )
    df["state_fips"] = df["fipsStateCode"].astype(str).str.zfill(2)

    # Parse dates
    for col in ["declarationDate", "incidentBeginDate", "incidentEndDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    # IA declared indicator (ihProgramDeclared or iaProgramDeclared)
    ia_col = "ihProgramDeclared" if "ihProgramDeclared" in df.columns else "iaProgramDeclared"
    df["ia_declared"] = df.get(ia_col, pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["pa_declared"] = df.get("paProgramDeclared", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # One row per disaster × county (some declarations have multiple program rows per county)
    key_cols = [
        "disasterNumber", "county_fips", "state_fips",
        "incidentBeginDate", "incidentEndDate", "declarationDate",
        "incidentType", "state",
    ]
    agg_cols = {"ia_declared": "max", "pa_declared": "max"}
    keep = [c for c in key_cols if c in df.columns]
    df_dedup = (
        df.groupby(keep, dropna=False)
        .agg(agg_cols)
        .reset_index()
    )
    df_dedup["event_year"] = df_dedup["incidentBeginDate"].dt.year

    print(f"  Unique disaster×county rows: {len(df_dedup):,}")
    print(f"  IA declared: {df_dedup['ia_declared'].sum():,} ({df_dedup['ia_declared'].mean():.1%})")
    return df_dedup


# ---------------------------------------------------------------------------
# 2b. Aggregate PA project costs → county-level damage estimate
# ---------------------------------------------------------------------------

def load_pa_damage() -> pd.DataFrame:
    path = RAW / "fema" / "pa_projects.parquet"
    require(path, "Run acquire/01_fema.py")

    pa = pd.read_parquet(path)
    print(f"  PA projects raw: {len(pa):,} rows")

    # Construct county FIPS
    state_col  = next((c for c in ["stateCode", "state", "stateNumberCode"] if c in pa.columns), None)
    county_col = next((c for c in ["countyCode", "designatedCounty", "countyFips"] if c in pa.columns), None)
    amount_col = next((c for c in ["projectAmount", "federalShareObligated", "totalObligated"] if c in pa.columns), None)
    disaster_col = "disasterNumber"

    if not all([state_col, county_col, amount_col]):
        print(f"  WARNING: Could not identify PA columns. Available: {list(pa.columns[:20])}")
        print("  Returning empty PA damage — per_capita_damage will be NaN")
        return pd.DataFrame(columns=["disasterNumber", "county_fips", "pa_damage_county"])

    pa["county_fips"] = (
        pa[state_col].astype(str).str.zfill(2) +
        pa[county_col].astype(str).str.zfill(3)
    )
    pa[amount_col] = pd.to_numeric(pa[amount_col], errors="coerce").fillna(0)

    damage = (
        pa.groupby([disaster_col, "county_fips"])[amount_col]
        .sum()
        .reset_index()
        .rename(columns={amount_col: "pa_damage_county"})
    )
    print(f"  PA damage: {len(damage):,} disaster×county aggregates")
    return damage


# ---------------------------------------------------------------------------
# 2c. Merge ACS population (vintage closest to but not exceeding event year)
# ---------------------------------------------------------------------------

def load_acs_population() -> pd.DataFrame:
    path = RAW / "census" / "acs_zcta_demographics.parquet"
    require(path, "Run acquire/04_census.py")

    acs = pd.read_parquet(path, columns=["zip code tabulation area", "B01003_001E", "acs_year"])
    acs = acs.rename(columns={
        "zip code tabulation area": "zcta5",
        "B01003_001E": "population",
    })
    acs["zcta5"] = acs["zcta5"].astype(str).str.zfill(5)
    acs["population"] = pd.to_numeric(acs["population"], errors="coerce")

    # For county-level population: sum ZCTAs in the county, weighted by primary assignment
    xwalk = pd.read_parquet(INTER / "zcta_county_primary.parquet")
    acs_county = (
        acs.merge(xwalk[["zcta5", "county_fips"]], on="zcta5", how="left")
        .groupby(["county_fips", "acs_year"])["population"]
        .sum()
        .reset_index()
        .rename(columns={"population": "county_population"})
    )
    return acs_county


def get_acs_vintage(event_year: int, acs_years: list) -> int:
    """Return the most recent ACS year that does not exceed event_year."""
    candidates = [y for y in acs_years if y <= event_year]
    return max(candidates) if candidates else min(acs_years)


# ---------------------------------------------------------------------------
# 2e. Historical IA exposure linkage (self-join on county)
# ---------------------------------------------------------------------------

def build_historical_ia_linkage(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each county × current event, find the most recent prior event in the
    same county and record:
      prior_disaster_number, prior_ia_received, prior_running_var_raw,
      prior_incidentType, years_since_prior, years_since_prior_bin
    """
    print("  Building historical IA exposure linkage...")

    df = events.sort_values(["county_fips", "incidentBeginDate"]).copy()
    df = df.reset_index(drop=True)

    # For each row, find the most recent prior row with same county
    results = []
    for county, grp in df.groupby("county_fips", sort=False):
        grp = grp.sort_values("incidentBeginDate").reset_index(drop=True)
        for i, row in grp.iterrows():
            prior_rows = grp[grp["incidentBeginDate"] < row["incidentBeginDate"]]
            if prior_rows.empty:
                results.append({
                    "disasterNumber":       row["disasterNumber"],
                    "county_fips":          county,
                    "prior_disaster_number": None,
                    "prior_ia_received":    None,
                    "prior_running_var_raw": None,
                    "prior_incident_type":  None,
                    "years_since_prior":    None,
                    "years_since_prior_bin": None,
                })
            else:
                prior = prior_rows.iloc[-1]
                days_diff = (row["incidentBeginDate"] - prior["incidentBeginDate"]).days
                years_diff = days_diff / 365.25
                bin_label = pd.cut(
                    [years_diff],
                    bins=TEMPORAL_BINS,
                    labels=TEMPORAL_LABELS,
                    right=True,
                )[0]
                results.append({
                    "disasterNumber":        row["disasterNumber"],
                    "county_fips":           county,
                    "prior_disaster_number": prior["disasterNumber"],
                    "prior_ia_received":     int(prior["ia_declared"]),
                    "prior_running_var_raw": prior.get("running_var_raw"),
                    "prior_incident_type":   prior.get("incidentType"),
                    "years_since_prior":     round(years_diff, 2),
                    "years_since_prior_bin": str(bin_label) if bin_label is not np.nan else None,
                })

    linkage = pd.DataFrame(results)
    n_with_prior = linkage["prior_disaster_number"].notna().sum()
    print(f"  {n_with_prior:,}/{len(linkage):,} county-events have a prior event")
    return linkage


# ---------------------------------------------------------------------------
# 2f. Political cycle variables
# ---------------------------------------------------------------------------

def build_political_vars(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add presidential election year flag, governor-president alignment,
    and county presidential vote margin.
    """
    # Presidential election year flag (pre-specified: event_year % 4 == 0)
    events["presidential_election_year"] = (events["event_year"] % 4 == 0).astype(int)

    # MIT Election Lab county returns
    mit_path = RAW / "elections" / "county_pres_returns.csv"
    if not mit_path.exists():
        print("  WARNING: MIT election returns not found — political vars will be partial")
        events["county_pres_vote_margin"] = None
        events["president_party"] = None
    else:
        mit = pd.read_csv(mit_path, low_memory=False)
        # Detect columns defensively
        year_col    = next((c for c in mit.columns if c.lower() == "year"), None)
        fips_col    = next((c for c in mit.columns if "fips" in c.lower() and "county" in c.lower()), None)
        party_col   = next((c for c in mit.columns if c.lower() == "party_simplified"), None)
        votes_col   = next((c for c in mit.columns if c.lower() in ("candidatevotes", "votes")), None)
        total_col   = next((c for c in mit.columns if "total" in c.lower() and "votes" in c.lower()), None)

        if year_col and fips_col and party_col and votes_col:
            mit = mit[mit[party_col].isin(["DEMOCRAT", "REPUBLICAN"])].copy()
            mit["county_fips"] = mit[fips_col].astype(str).str.zfill(5)
            mit["vote_share"] = pd.to_numeric(mit[votes_col], errors="coerce")
            if total_col:
                mit["vote_share"] = mit["vote_share"] / pd.to_numeric(mit[total_col], errors="coerce")

            wide = mit.pivot_table(
                index=["county_fips", year_col],
                columns=party_col,
                values="vote_share" if total_col else votes_col,
                aggfunc="sum",
            ).reset_index()
            wide.columns.name = None
            wide = wide.rename(columns={year_col: "election_year"})
            if "REPUBLICAN" in wide.columns and "DEMOCRAT" in wide.columns:
                wide["county_pres_vote_margin"] = (
                    wide.get("REPUBLICAN", 0) - wide.get("DEMOCRAT", 0)
                )
            wide["election_year"] = pd.to_numeric(wide["election_year"], errors="coerce")

            # Merge: for each event, use most recent prior election year
            def get_prior_election_year(yr):
                elec_years = sorted(wide["election_year"].dropna().unique())
                candidates = [y for y in elec_years if y <= yr]
                return max(candidates) if candidates else None

            events["prior_election_year"] = events["event_year"].apply(get_prior_election_year)
            events = events.merge(
                wide[["county_fips", "election_year", "county_pres_vote_margin"]],
                left_on=["county_fips", "prior_election_year"],
                right_on=["county_fips", "election_year"],
                how="left",
            ).drop(columns=["election_year"], errors="ignore")

            # President party from national election result
            nat = mit.groupby([year_col, party_col])[votes_col].sum().reset_index()
            nat["vote_share"] = nat[votes_col] / nat.groupby(year_col)[votes_col].transform("sum")
            winner = nat.loc[nat.groupby(year_col)["vote_share"].idxmax()]
            pres_party = winner.set_index(year_col)[party_col].to_dict()
            events["president_party"] = events["prior_election_year"].map(pres_party)
        else:
            print(f"  WARNING: Unexpected MIT election columns: {list(mit.columns[:10])}")
            events["county_pres_vote_margin"] = None
            events["president_party"] = None

    # Klarner governor party
    klarner_path = RAW / "elections" / "klarner_state_politics.csv"
    if not klarner_path.exists():
        print("  WARNING: Klarner data not found — governor party unavailable")
        events["governor_party"] = None
        events["governor_president_alignment"] = None
    else:
        gov = pd.read_csv(klarner_path, low_memory=False)
        # Detect columns defensively — Klarner column names vary by download
        year_col = next((c for c in gov.columns if c.lower() in ("year", "stateyear")), None)
        state_col = next((c for c in gov.columns if c.lower() in ("state", "statename", "stateabbr", "stateid")), None)
        gov_col = next((c for c in gov.columns if "gov" in c.lower() and "party" in c.lower()), None)
        if gov_col is None:
            gov_col = next((c for c in gov.columns if "gov" in c.lower()), None)

        if year_col and gov_col and state_col:
            gov = gov[[state_col, year_col, gov_col]].rename(columns={
                state_col: "state_key",
                year_col:  "gov_year",
                gov_col:   "governor_party_raw",
            })
            gov["gov_year"] = pd.to_numeric(gov["gov_year"], errors="coerce")
            # Standardize governor party to R/D
            gov["governor_party"] = gov["governor_party_raw"].astype(str).str.upper().str[0]
            gov["governor_party"] = gov["governor_party"].where(gov["governor_party"].isin(["R", "D"]))

            events = events.merge(
                gov[["state_key", "gov_year", "governor_party"]],
                left_on=["state", "event_year"],
                right_on=["state_key", "gov_year"],
                how="left",
            ).drop(columns=["state_key", "gov_year"], errors="ignore")

            if "president_party" in events.columns and "governor_party" in events.columns:
                pres = events["president_party"].str.upper().str[0]
                gov_p = events["governor_party"].str.upper().str[0]
                events["governor_president_alignment"] = (pres == gov_p).astype("Int8")
        else:
            print(f"  WARNING: Could not detect Klarner columns: {list(gov.columns[:10])}")
            events["governor_party"] = None
            events["governor_president_alignment"] = None

    return events


# ---------------------------------------------------------------------------
# 2g. Hazard type classification from NOAA
# ---------------------------------------------------------------------------

def classify_hazard_types(events: pd.DataFrame) -> pd.DataFrame:
    noaa_path = RAW / "noaa" / "storm_events_details.parquet"
    require(noaa_path, "Run acquire/02_noaa_storm_events.py")

    noaa = pd.read_parquet(
        noaa_path,
        columns=["STATE_FIPS", "CZ_FIPS", "CZ_TYPE", "EVENT_TYPE",
                 "BEGIN_YEARMONTH", "BEGIN_DAY", "END_YEARMONTH", "END_DAY"],
    )
    # County events only
    noaa = noaa[noaa["CZ_TYPE"] == "C"].copy()
    noaa["county_fips"] = (
        noaa["STATE_FIPS"].astype(str).str.zfill(2) +
        noaa["CZ_FIPS"].astype(str).str.zfill(3)
    )

    # Parse NOAA dates
    def parse_noaa_date(ym, day):
        try:
            return pd.to_datetime(f"{int(ym)}{int(day):02d}", format="%Y%m%d", errors="coerce")
        except Exception:
            return pd.NaT

    noaa["noaa_begin"] = pd.to_datetime(
        noaa["BEGIN_YEARMONTH"].astype(str) + noaa["BEGIN_DAY"].astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce",
    )
    noaa["noaa_end"] = pd.to_datetime(
        noaa["END_YEARMONTH"].astype(str) + noaa["END_DAY"].astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce",
    )

    # Flood-related event types
    flood_types    = {"Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood", "Storm Surge/Tide"}
    hurricane_types = {"Hurricane", "Hurricane (Typhoon)", "Tropical Storm"}

    results = []
    for _, row in events.iterrows():
        if pd.isnull(row.get("incidentBeginDate")):
            results.append("unknown")
            continue

        begin = row["incidentBeginDate"]
        end   = row.get("incidentEndDate", begin)
        county = row["county_fips"]

        # Match NOAA events: same county, overlapping date window (±3 days)
        window_start = begin - pd.Timedelta(days=NOAA_MATCH_DAYS)
        window_end   = (end if pd.notna(end) else begin) + pd.Timedelta(days=NOAA_MATCH_DAYS)

        matched = noaa[
            (noaa["county_fips"] == county) &
            (noaa["noaa_begin"] <= window_end) &
            (noaa["noaa_end"]   >= window_start)
        ]

        if matched.empty:
            results.append("unknown")
            continue

        types = set(matched["EVENT_TYPE"].dropna().unique())
        has_flood     = bool(types & flood_types)
        has_hurricane = bool(types & hurricane_types)
        n_types       = len(types)

        if has_hurricane:
            hazard = "hurricane"
        elif has_flood and n_types == 1:
            hazard = "flood_only"
        elif has_flood:
            hazard = "flood_mixed"
        elif n_types == 1:
            hazard = "severe_storm"
        else:
            hazard = "mixed"

        results.append(hazard)

    events["hazard_type"] = results
    print("  Hazard type distribution:")
    for ht, cnt in events["hazard_type"].value_counts().items():
        print(f"    {ht}: {cnt:,}")
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = INTER / "events_county_panel.parquet"
    if out.exists():
        print("events_county_panel.parquet already exists — delete to rebuild")
        df = pd.read_parquet(out)
        print(f"  {len(df):,} rows, columns: {list(df.columns)}")
        return

    print("=== Step 2: Construct Events ===\n")

    require(INTER / "zcta_county_primary.parquet", "Run 01_standardize_geography.py")
    require(INTER / "fips_standardized.parquet",   "Run 01_standardize_geography.py")

    # 2a. Load and reshape declarations
    print("2a. Loading FEMA declarations...")
    events = load_fema_declarations()

    # 2b. PA damage aggregation
    print("\n2b. Aggregating PA damage...")
    pa_damage = load_pa_damage()
    events = events.merge(pa_damage, on=["disasterNumber", "county_fips"], how="left")
    events["pa_damage_county"] = events["pa_damage_county"].fillna(0)

    # 2c. ACS population + per-capita damage
    print("\n2c. Merging ACS population...")
    acs_county = load_acs_population()
    acs_years = sorted(acs_county["acs_year"].unique())
    events["acs_vintage"] = events["event_year"].apply(
        lambda y: get_acs_vintage(y, acs_years) if pd.notna(y) else None
    )
    events = events.merge(
        acs_county, left_on=["county_fips", "acs_vintage"],
        right_on=["county_fips", "acs_year"], how="left",
    ).drop(columns=["acs_year"], errors="ignore")
    events["per_capita_damage"] = (
        events["pa_damage_county"] / events["county_population"].replace(0, float("nan"))
    )
    # Placeholder running variable (TTR-adjusted in 04b)
    events["running_var_raw"] = events["per_capita_damage"]

    print(f"  per_capita_damage: median={events['per_capita_damage'].median():.2f}, "
          f"nonzero={events['per_capita_damage'].gt(0).sum():,}")

    # 2e. Historical IA linkage
    print("\n2e. Building historical IA linkage...")
    linkage = build_historical_ia_linkage(events)
    events = events.merge(linkage, on=["disasterNumber", "county_fips"], how="left")

    # 2f. Political cycle variables
    print("\n2f. Adding political cycle variables...")
    events = build_political_vars(events)

    # 2g. Hazard type classification
    print("\n2g. Classifying hazard types...")
    events = classify_hazard_types(events)

    events.to_parquet(out, index=False)
    print(f"\nSaved {len(events):,} rows → {out.relative_to(ROOT)}")
    print(f"Columns: {list(events.columns)}")

    # Validation
    print("\n--- Validation ---")
    print(f"  Year range: {events['event_year'].min():.0f} – {events['event_year'].max():.0f}")
    print(f"  IA declared: {events['ia_declared'].mean():.1%}")
    n_with_prior = events["prior_disaster_number"].notna().sum()
    print(f"  Has prior event: {n_with_prior:,}/{len(events):,}")
    print(f"  running_var_raw symmetric check: "
          f"mean={events['running_var_raw'].mean():.3f} "
          f"(should not be heavily skewed around 0 after TTR adjustment)")


def get_acs_vintage(event_year, acs_years):
    candidates = [y for y in acs_years if y <= event_year]
    return max(candidates) if candidates else min(acs_years)


if __name__ == "__main__":
    main()

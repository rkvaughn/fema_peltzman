# FEMA Individual Assistance and Household Mitigation: Do Disaster Aid Expectations Alter Private Loss Prevention Behavior?

## Research Question

Does exposure to FEMA Individual Assistance (IA) grants alter household pre-disaster mitigation effort and post-disaster claiming behavior? The direction of the effect is an open empirical question with two competing theoretical predictions:

**Peltzman/substitution hypothesis:** The safety net of expected federal aid lowers the private cost of property damage, so households **reduce** physical mitigation effort (boarding windows, sandbagging, securing utilities, moving valuables) and shift toward rapid evacuation, letting the government cover losses.

**Complementarity hypothesis (Andor et al. 2020; Botzen et al. 2019):** Households view physical mitigation and expected aid as complements rather than substitutes. Knowing that aid is available regardless of mitigation effort, households face lower opportunity cost of mitigation time (no need to focus on insurance paperwork or financial planning) and may actually **increase** physical protection of property. Evidence from German flood-prone households and Hurricane Sandy homeowners in New York supports this direction for non-financial protection measures, even as insurance take-up shows clear substitution.

The sign of the treatment effect — whether expectations of federal aid crowd out or crowd in private mitigation — reveals which institutional features of the U.S. disaster aid system dominate household decision-making. A null result is also informative.

This project investigates two distinct causal mechanisms:

**Mechanism A — Learned Expectations (Pre-Disaster Behavior):**
Households in areas that received IA in prior disasters have updated expectations of future federal assistance. When a new hazard approaches, these households may invest more or less in last-minute property hardening depending on whether the substitution or complementarity channel dominates. This is identified using historical IA receipt, instrumented by the prior event's per-capita damage threshold, as a source of exogenous variation in current expectations.

**Mechanism B — Contemporaneous Moral Hazard (Post-Disaster Behavior):**
After an IA declaration, households in declared areas change their post-disaster behavior — filing more claims, reporting higher damage, repairing less out-of-pocket, and substituting IA for private insurance claims. This is identified using the contemporaneous per-capita damage threshold in a standard fuzzy RD design, which is valid for post-disaster outcomes because the declaration has already occurred when households make claiming decisions. Note that the IA declaration is best understood as a *signal* of broader federal support — Deryugina (2017) shows that non-disaster government transfers (UI, Medicaid, income maintenance) after hurricanes significantly exceed direct disaster aid — so the estimated effect captures the full signaling value of being in a declared disaster area, not just the IA grant itself.

These are complementary analyses. Mechanism A tests whether expectations of aid alter effort before the storm hits. Mechanism B tests whether actual aid availability changes behavior after it hits. Together they bracket the full moral hazard channel.

---

## Identification Strategy

### Mechanism A: Historical IV for Pre-Disaster Mitigation

#### The Timeline Problem

The original design proposed using the contemporaneous per-capita damage threshold as an instrument for IA availability, with pre-disaster mitigation behavior as the outcome. This has a fatal chronological flaw: at the moment households decide whether to board windows or sandbag (t = -1), the final county damage tally is unknown. Households above and below the eventual threshold hold identical expectations of IA receipt. Any effect detected at the contemporaneous threshold for pre-disaster outcomes would reflect unobserved hazard intensity, not moral hazard.

#### Revised Design

Instead, we exploit **historical IA declarations** as the source of variation in household expectations. The identifying assumption is that households in ZIPs that narrowly qualified for IA in a prior disaster event (t-k) update their subjective probability of future IA receipt upward, relative to households in ZIPs that narrowly missed the threshold in the same prior event.

**First stage (historical threshold → prior IA receipt):**

```
IA_{i,t-k} = α + f(RunningVar_{c,t-k}) + γ · 1[RunningVar_{c,t-k} > 0] + ε_{i,t-k}
```

Where `IA_{i,t-k}` is an indicator for ZIP i receiving IA in the most recent prior event t-k, `RunningVar_{c,t-k}` is the TTR-adjusted county-level per-capita damage minus the published threshold in year t-k, and `f(·)` is a flexible polynomial. The coefficient γ estimates the first-stage discontinuity in prior IA receipt at the historical threshold.

**Second stage (predicted prior IA → current mitigation):**

```
Mitigation_{i,t} = β₀ + β₁ · IA_hat_{i,t-k} + X_{i,t}δ + HazardIntensity_{i,t}λ
                   + WarningLeadTime_{i,t}θ + η_{i,t}
```

If the Peltzman/substitution effect holds, β₁ < 0: areas conditioned by historical IA receipt invest less in pre-disaster mitigation during the current event. If the complementarity effect holds (Andor et al. 2020), β₁ > 0: prior IA receipt increases mitigation effort. The sign of β₁ is the primary empirical question. All hypothesis tests are two-sided.

**Key identifying assumptions:**
1. Historical threshold crossing is as-good-as-random conditional on the running variable (standard RD assumption for the prior event)
2. The historical threshold affects current mitigation only through updated IA expectations, not through other channels (exclusion restriction)
3. Households generalize from past IA experience to form expectations about future events

**Threats to identification:**
- Prior IA receipt may correlate with unobserved community resilience characteristics that persist across events
- The relevance of the historical IV depends on temporal proximity and hazard-type similarity between events t-k and t
- ZIPs with prior IA may have different post-disaster infrastructure (rebuilt to code, new drainage) that mechanically changes damage patterns independent of behavioral mitigation

#### Media Interaction Specification

To further isolate the expectations mechanism, we interact historical IA receipt with a pre-event media intensity index:

```
Mitigation_{i,t} = γ₀ + γ₁ · IA_hat_{i,t-k} + γ₂ · Media_{c,t}
                   + γ₃ · IA_hat_{i,t-k} × Media_{c,t} + X_{i,t}δ + ε_{i,t}
```

We hypothesize γ₃ ≠ 0: when media coverage is high (signaling a major, likely-to-be-declared event), households with IA history respond differently than those without. Under the substitution hypothesis, γ₃ < 0 (IA-experienced households abandon mitigation fastest when media signals a big event). Under complementarity, γ₃ > 0 (media attention activates learned mitigation routines). Media intensity can be constructed from GDELT event counts or news API data for the county in the 72 hours prior to landfall/event onset. Gallagher (2014) found that media exposure drives risk updating even in non-flooded communities within the same television market, supporting the relevance of this channel. This is a secondary specification — data availability for a clean media index is uncertain (confidence: 4/10).

### Mechanism B: Contemporaneous RD for Post-Disaster Outcomes

The standard fuzzy RD using the contemporaneous per-capita damage threshold remains valid for outcomes measured **after** the IA declaration has been issued and is public knowledge. At this point, households in declared areas know they are eligible for IA; those in non-declared areas know they are not.

**Running variable:** TTR-adjusted per-capita estimated damage at the county/state level relative to the FEMA IA recommendation threshold for the current event.

**Instrument:** Threshold crossing → IA declaration → post-disaster behavior.

**Outcomes (all measured post-declaration):**
- IA application rate per housing unit
- IA approval rate and average award amount
- NFIP claim severity relative to hazard intensity
- Contents-to-structural damage ratio (post-disaster reporting, not pre-disaster mitigation)
- SBA disaster loan application rate (substitution for private savings)
- Time from event to claim filing

### TTR-Adjusted Running Variable

FEMA's recommendation algorithm heavily weights a state's **Total Taxable Resources (TTR)** when evaluating disaster declarations. High-TTR states face a higher effective threshold than low-TTR states. Failing to adjust the running variable for state fiscal capacity introduces selection bias and weakens the first stage.

**Construction:**

```
AdjustedRunningVar_{c,t} = (PerCapitaDamage_{c,t} × TTR_adjustment_{s,t}) - Threshold_{t}
```

Where `TTR_adjustment_{s,t}` normalizes each state's fiscal capacity relative to the national median. The exact functional form of FEMA's TTR weighting is not publicly documented — we test linear, log, and percentile-rank specifications and report robustness across all three. Confidence in being able to reconstruct the correct adjustment: 5/10. This is an important data gap to resolve early.

### Unit of Analysis

**ZIP code (5-digit).** FEMA claims data is natively reported at the ZIP level, avoiding crosswalk-induced measurement error in the outcome variable. Census demographics are available at the ZCTA level via ACS, mapping nearly 1:1 to USPS ZIP codes.

### Key Endogeneity Concerns

1. **Media-driven panic:** Major declarations get more news coverage, which independently affects evacuation behavior. The historical IV (Mechanism A) addresses this for pre-disaster outcomes. For post-disaster outcomes (Mechanism B), media intensity enters as a control.
2. **State/local emergency declarations:** States may issue their own declarations below the federal threshold, providing alternative aid. Track state-level declarations as controls.
3. **Repeat disaster exposure:** ZIPs with prior IA experience may behave differently through channels other than expectations (rebuilt infrastructure, changed drainage, etc.). Include prior declaration history and prior damage severity as covariates. Test whether the Mechanism A effect varies by years-since-prior-event.
4. **Threshold manipulation:** FEMA or states could game damage estimates to cross the threshold. Test with McCrary density test on the running variable AND placebo RDs using predetermined covariates (median household income, presidential vote margin) as outcomes.
5. **TTR endogeneity:** States with high fiscal capacity may have systematically different emergency management infrastructure. The TTR adjustment partially addresses this, but state fixed effects are also needed.
6. **Political cycle contamination (Schneider and Kunze 2025):** Political bias in disaster declarations is hump-shaped in disaster severity — concentrated in medium-intensity events where the per-capita threshold operates. Areas governed by the president's co-partisans receive up to twice as many declarations for medium-intensity hurricanes. This means the first stage may partially capture political sorting rather than threshold quasi-randomness. Controls: include presidential election year indicator, governor-president party alignment indicator, and county-level presidential vote margin. Test for differential first-stage strength in election years vs. non-election years. If the first stage is substantially stronger in election years, political sorting is likely contaminating the design. Additional robustness: restrict primary sample to flood events, which are less politically salient than hurricanes (the Schneider-Kunze finding is hurricane-specific).

---

## Outcome Variables

### Mechanism A Outcomes: Pre-Disaster Mitigation Behavior

| Variable | Proxy For | Source | Confidence |
|---|---|---|---|
| Contents-to-structural damage ratio | Whether occupants moved belongings before impact | NFIP claims | 6/10 |
| Damage per foot of flood depth | Residual damage after controlling for hazard intensity; lower = more mitigation | NFIP claims + USGS HWM | 5/10 |
| Total claim amount relative to hazard intensity | Residual unexplained damage | NFIP + IA claims | 5/10 |
| Claim denial rate for "insufficient damage" | Marginal events where mitigation made the difference | OpenFEMA IA claims | 4/10 |

**Critical confound control:** All Mechanism A outcomes must be interacted with **warning lead time** (hours of advance notice from NWS). High contents damage in a flash event reflects physics, not moral hazard. The Peltzman calculus requires time to execute. Restrict the primary Mechanism A sample to events with ≥24 hours of advance warning.

**Tenure heterogeneity:** Interact all Mechanism A outcomes with ZCTA-level homeownership rate (ACS B25003). Renters have zero incentive to mitigate structural damage but strong incentive to protect contents. Owners face the opposite tradeoff. The Peltzman mechanism operates differently across tenure status — renters are only eligible for Other Needs Assistance (ONA), while owners qualify for full IA housing assistance.

### Mechanism B Outcomes: Post-Disaster Claiming Behavior

| Variable | Proxy For | Source | Confidence |
|---|---|---|---|
| IA application rate per housing unit | Willingness to seek federal aid | OpenFEMA IA + ACS | 8/10 |
| IA approval rate | Declaration effect on federal transfers | OpenFEMA IA | 7/10 |
| Average IA award amount | Magnitude of moral hazard dollar impact | OpenFEMA IA | 7/10 |
| NFIP claim severity relative to hazard | Whether damage reporting inflates post-declaration | NFIP claims | 6/10 |
| SBA disaster loan application rate | Substitution between federal aid channels | SBA disaster loans | 5/10 |
| Contents-to-structural damage ratio (post-declaration) | Reporting behavior, not mitigation | NFIP claims | 6/10 |
| Time from event to IA application filing | Speed of aid-seeking | OpenFEMA IA timestamps | 5/10 |

---

## Data Sources and Acquisition Plan

### Phase 1: Core FEMA Data

All OpenFEMA datasets are available via REST API at `https://www.fema.gov/api/open/`.

#### 1A. Disaster Declarations Summaries (v2)

```
Endpoint: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries
```

**Purpose:** Identify all disaster events, their declaration status, IA designation, and geographic scope.

**Key fields:**
- `disasterNumber` — unique event ID
- `state`, `fipsStateCode`, `fipsCountyCode` — geography
- `declarationType` (DR, EM, FM) — filter to DR (major disaster)
- `ihProgramDeclared` — boolean, whether IA (Individual & Households) was declared
- `incidentBeginDate`, `incidentEndDate` — event timing
- `incidentType` — hazard type (Hurricane, Flood, Severe Storm, etc.)
- `designatedArea` — county-level designation

**Download instructions for Claude Code agent:**

```bash
# Download full dataset (paginated, 1000 records per page)
mkdir -p data/raw/fema

# Get total count first
curl -s "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?\$count=true&\$top=1" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['metadata']['count'])"

# Download all records
python3 << 'EOF'
import requests
import pandas as pd
import time

base_url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
all_records = []
skip = 0
batch_size = 1000

while True:
    params = {
        "$top": batch_size,
        "$skip": skip,
        "$orderby": "disasterNumber asc"
    }
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("DisasterDeclarationsSummaries", [])
    if not records:
        break
    all_records.extend(records)
    print(f"  Downloaded {len(all_records)} records...")
    skip += batch_size
    time.sleep(0.5)

df = pd.DataFrame(all_records)
df.to_parquet("data/raw/fema/disaster_declarations.parquet", index=False)
df.to_csv("data/raw/fema/disaster_declarations.csv", index=False)
print(f"Total records: {len(df)}")
print(f"Date range: {df['incidentBeginDate'].min()} to {df['incidentBeginDate'].max()}")
print(f"IA declarations: {df['ihProgramDeclared'].sum()}")
EOF
```

#### 1B. Individual Assistance Applications (v1) — FIMA Dataset

```
Endpoint: https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters
         https://www.fema.gov/api/open/v1/HousingAssistanceOwners
         https://www.fema.gov/api/open/v1/HousingAssistanceRenters
```

**Purpose:** ZIP-level IA application and award data. Data is natively at the ZIP level — no crosswalk needed. Critical for both Mechanism A (prior IA receipt as treatment history) and Mechanism B (post-declaration claiming behavior as outcome).

**Key fields:**
- `disasterNumber` — links to declarations
- `damagedZipCode`, `county` — geography (ZIP-level, native unit of analysis)
- `totalApprovedIhpAmount` — approved IA amount
- `repairAmount`, `replacementAmount` — structural damage indicators
- `rentalAmount`, `otherNeedsAmount` — displacement indicators
- `floodDamageAmount`, `foundationDamageAmount`, `roofDamageAmount` — damage specifics
- `habitabilityRepairsRequired` — severity flag
- `rpfvl` (Real Property Fair Value Loss) — proportional damage

**Download instructions:**

```bash
python3 << 'EOF'
import requests
import pandas as pd
import time

datasets = {
    "housing_owners": "https://www.fema.gov/api/open/v1/HousingAssistanceOwners",
    "housing_renters": "https://www.fema.gov/api/open/v1/HousingAssistanceRenters",
    "ia_large_disasters": "https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters"
}

for name, base_url in datasets.items():
    print(f"\nDownloading {name}...")
    all_records = []
    skip = 0
    batch_size = 1000

    while True:
        params = {"$top": batch_size, "$skip": skip}
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Error at skip={skip}: {e}. Retrying...")
            time.sleep(5)
            continue

        data = resp.json()
        key = [k for k in data.keys() if k != "metadata"][0] if "metadata" in data else list(data.keys())[0]
        records = data.get(key, [])
        if not records:
            break
        all_records.extend(records)
        print(f"  {name}: {len(all_records)} records...")
        skip += batch_size
        time.sleep(0.5)

    df = pd.DataFrame(all_records)
    df.to_parquet(f"data/raw/fema/{name}.parquet", index=False)
    print(f"  {name}: {len(df)} total records saved")
EOF
```

#### 1C. NFIP Claims (OpenFEMA)

```
Endpoint: https://www.fema.gov/api/open/v1/FimaNfipClaims
```

**Purpose:** Flood insurance claims with detailed damage breakdowns. Critical for measuring mitigation behavior via structural-vs-contents damage ratios.

**Key fields:**
- `dateOfLoss` — event timing
- `reportedZipCode`, `countyCode`, `state` — geography
- `amountPaidOnBuildingClaim`, `amountPaidOnContentsClaim` — damage split
- `totalBuildingInsuranceCoverage`, `totalContentsInsuranceCoverage` — coverage levels
- `waterDepth`, `numberOfFloorsInInsuredBuilding` — hazard intensity at property
- `elevatedBuildingIndicator`, `baseFloodElevation` — mitigation characteristics
- `primaryResidence` — occupancy filter

**Download instructions:**

```bash
python3 << 'EOF'
import requests
import pandas as pd
import time

base_url = "https://www.fema.gov/api/open/v1/FimaNfipClaims"
all_records = []
skip = 0
batch_size = 1000

while True:
    params = {"$top": batch_size, "$skip": skip}
    try:
        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error at skip={skip}: {e}. Retrying in 10s...")
        time.sleep(10)
        continue

    data = resp.json()
    key = [k for k in data.keys() if k != "metadata"][0]
    records = data.get(key, [])
    if not records:
        break
    all_records.extend(records)
    if len(all_records) % 50000 == 0:
        print(f"  NFIP Claims: {len(all_records)} records...")
    skip += batch_size
    time.sleep(0.3)

df = pd.DataFrame(all_records)
df.to_parquet("data/raw/fema/nfip_claims.parquet", index=False)
print(f"NFIP Claims: {len(df)} total records")
EOF
```

#### 1D. FEMA Public Assistance (for per-capita damage running variable)

```
Endpoint: https://www.fema.gov/api/open/v1/PublicAssistanceApplicants
          https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsSummaries
```

**Purpose:** Public Assistance project costs are used by FEMA to calculate per-capita damage estimates — which drive the IA threshold. These reconstruct the running variable.

```bash
python3 << 'EOF'
import requests
import pandas as pd
import time

datasets = {
    "pa_applicants": "https://www.fema.gov/api/open/v1/PublicAssistanceApplicants",
    "pa_projects": "https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsSummaries"
}

for name, base_url in datasets.items():
    print(f"\nDownloading {name}...")
    all_records = []
    skip = 0
    while True:
        params = {"$top": 1000, "$skip": skip}
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
        except:
            time.sleep(5)
            continue
        data = resp.json()
        key = [k for k in data.keys() if k != "metadata"][0]
        records = data.get(key, [])
        if not records:
            break
        all_records.extend(records)
        if len(all_records) % 10000 == 0:
            print(f"  {len(all_records)} records...")
        skip += 1000
        time.sleep(0.3)

    df = pd.DataFrame(all_records)
    df.to_parquet(f"data/raw/fema/{name}.parquet", index=False)
    print(f"  {name}: {len(df)} total records")
EOF
```

### Phase 2: Hazard Exposure Data

#### 2A. NOAA Storm Events Database

```
Source: https://www.ncdc.noaa.gov/stormevents/ftp.jsp
FTP:    https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
```

**Purpose:** Independent hazard intensity measures at the county level. Needed to verify that events just above/below the FEMA threshold have comparable actual severity.

**Key files:** `StormEvents_details-ftp_v1.0_dYYYY.csv.gz` for each year.

**Key fields:**
- `BEGIN_DATE_TIME`, `END_DATE_TIME` — event timing
- `STATE_FIPS`, `CZC_FIPS` — county geography
- `EVENT_TYPE` — hazard classification
- `DAMAGE_PROPERTY`, `DAMAGE_CROPS` — dollar damage estimates
- `DEATHS_DIRECT`, `INJURIES_DIRECT` — severity measures
- `FLOOD_CAUSE`, `TOR_F_SCALE`, `MAGNITUDE` — intensity metrics
- `BEGIN_LAT`, `BEGIN_LON` — point locations

```bash
mkdir -p data/raw/noaa

# Download all available storm events detail files
python3 << 'EOF'
import requests
from bs4 import BeautifulSoup
import os
import time

base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
resp = requests.get(base_url)
soup = BeautifulSoup(resp.text, "html.parser")

links = [a["href"] for a in soup.find_all("a") if "details" in a.get("href", "").lower() and a["href"].endswith(".gz")]

for link in links:
    fname = link.split("/")[-1]
    outpath = f"data/raw/noaa/{fname}"
    if os.path.exists(outpath):
        continue
    print(f"Downloading {fname}...")
    r = requests.get(base_url + link, stream=True)
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    time.sleep(0.5)

print("Done downloading storm events.")
EOF

# Combine into single parquet
python3 << 'EOF'
import pandas as pd
import glob

files = sorted(glob.glob("data/raw/noaa/StormEvents_details*.csv.gz"))
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, compression="gzip", low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet("data/raw/noaa/storm_events_details.parquet", index=False)
print(f"Combined storm events: {len(combined)} records, {combined['YEAR'].min()}-{combined['YEAR'].max()}")
EOF
```

#### 2B. USGS Flood Event Viewer / National Water Model

```
Source: https://stn.wim.usgs.gov/STNServices/
API:    https://stn.wim.usgs.gov/STNServices/Events.json
        https://stn.wim.usgs.gov/STNServices/HWMs.json (high water marks)
        https://stn.wim.usgs.gov/STNServices/Sites.json
```

**Purpose:** Objective flood depth measurements at specific locations. Provides ground-truth hazard intensity independent of FEMA damage estimates.

```bash
mkdir -p data/raw/usgs

python3 << 'EOF'
import requests
import pandas as pd

events = requests.get("https://stn.wim.usgs.gov/STNServices/Events.json").json()
pd.DataFrame(events).to_parquet("data/raw/usgs/stn_events.parquet", index=False)
print(f"STN Events: {len(events)}")

hwms = requests.get("https://stn.wim.usgs.gov/STNServices/HWMs.json").json()
pd.DataFrame(hwms).to_parquet("data/raw/usgs/high_water_marks.parquet", index=False)
print(f"High Water Marks: {len(hwms)}")

sites = requests.get("https://stn.wim.usgs.gov/STNServices/Sites.json").json()
pd.DataFrame(sites).to_parquet("data/raw/usgs/stn_sites.parquet", index=False)
print(f"STN Sites: {len(sites)}")
EOF
```

#### 2C. NOAA National Hurricane Center HURDAT2

```
Source: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt
```

**Purpose:** Wind speed and track data for hurricanes. Allows computing wind exposure at the ZIP centroid level for hurricane events.

```bash
mkdir -p data/raw/nhc
curl -o data/raw/nhc/hurdat2.txt \
  "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"

python3 << 'EOF'
import pandas as pd

records = []
current_storm = None

with open("data/raw/nhc/hurdat2.txt") as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) == 4:
            current_storm = {"id": parts[0], "name": parts[1], "entries": int(parts[2])}
        elif len(parts) >= 8:
            records.append({
                "storm_id": current_storm["id"],
                "storm_name": current_storm["name"],
                "date": parts[0],
                "time": parts[1],
                "record_id": parts[2],
                "status": parts[3],
                "lat": parts[4],
                "lon": parts[5],
                "max_wind_kt": int(parts[6]) if parts[6] else None,
                "min_pressure_mb": int(parts[7]) if parts[7] else None
            })

df = pd.DataFrame(records)
df.to_parquet("data/raw/nhc/hurdat2.parquet", index=False)
print(f"HURDAT2: {len(df)} track points, {df['storm_id'].nunique()} storms")
EOF
```

#### 2D. NOAA National Weather Service Warning Archives

```
Source: https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml
        (Iowa Environmental Mesonet — IEM — archives all NWS warnings with geometries)
API:    https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py
```

**Purpose:** Warning lead time data. Critical for Mechanism A — must distinguish "didn't mitigate because of moral hazard" from "couldn't mitigate because the event arrived with <2 hours notice." The Peltzman calculus requires time to execute a mitigation decision.

**Key fields:**
- `ISSUED` — warning issuance datetime
- `EXPIRED` — warning expiration datetime
- `PHENOM` — phenomenon type (FF=Flash Flood, FL=Flood, HU=Hurricane, TO=Tornado, SV=Severe Thunderstorm)
- `SIG` — significance (W=Warning, A=Watch, Y=Advisory)
- `WFO` — issuing Weather Forecast Office
- `STATUS` — NEW, CON (continued), CAN (cancelled), EXP (expired)
- Polygon geometry — spatial extent of warning

**Download instructions:**

```bash
mkdir -p data/raw/nws

# IEM provides bulk download of NWS warnings with geometries
# Download as shapefiles by year
python3 << 'EOF'
import requests
import os
import time

base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"

# Download warnings for each year — focus on flood, hurricane, severe storm warnings
# IEM supports bulk CSV download with parameters
for year in range(2000, 2026):
    outpath = f"data/raw/nws/warnings_{year}.csv"
    if os.path.exists(outpath):
        continue

    print(f"Downloading NWS warnings {year}...")
    params = {
        "year1": year,
        "month1": 1,
        "day1": 1,
        "year2": year,
        "month2": 12,
        "day2": 31,
        "fmt": "csv",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=300)
        resp.raise_for_status()
        with open(outpath, "w") as f:
            f.write(resp.text)
        print(f"  {year}: {len(resp.text)} bytes")
    except Exception as e:
        print(f"  {year}: FAILED — {e}")
    time.sleep(2)

# Combine into single dataset
import pandas as pd
import glob

files = sorted(glob.glob("data/raw/nws/warnings_*.csv"))
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet("data/raw/nws/nws_warnings.parquet", index=False)
    print(f"Combined NWS warnings: {len(combined)} records")
else:
    print("WARNING: No NWS warning files successfully downloaded.")
    print("Alternative: Download directly from IEM at https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml")
EOF
```

**Derived variable: Warning Lead Time**

For each disaster event × county, compute warning lead time as: `incidentBeginDate - earliest_warning_issued` for relevant phenomena (FL, FF, HU, TO, SV with SIG=W). This produces hours of advance warning at the county level. Events with <6 hours of warning are classified as "flash" events. Primary Mechanism A analysis restricted to events with ≥24 hours warning; robustness tested across 6h, 12h, 24h, 48h thresholds.

### Phase 3: Census Demographics

#### 3A. American Community Survey 5-Year Estimates (ZCTA Level)

```
API: https://api.census.gov/data/{year}/acs/acs5
```

**Purpose:** ZCTA-level demographics for matching and controls. ZCTAs map nearly 1:1 to USPS ZIP codes, so merging with FEMA claims requires only a simple join on the 5-digit code.

**Key variables (ACS table codes):**
- `B01003_001E` — Total population
- `B19013_001E` — Median household income
- `B25077_001E` — Median home value
- `B25003_001E`, `B25003_002E`, `B25003_003E` — Total, owner-occupied, renter-occupied housing units
- `B25034_001E` through `B25034_011E` — Year structure built (housing age distribution)
- `B25024_001E` through `B25024_011E` — Units in structure (SFH vs. MFH)
- `B01002_001E` — Median age
- `B02001_001E` through `B02001_006E` — Race
- `B25002_001E`, `B25002_002E`, `B25002_003E` — Occupancy status
- `B25014_001E` through `B25014_007E` — Overcrowding (persons per room)

```bash
mkdir -p data/raw/census

python3 << 'EOF'
import requests
import pandas as pd
import time
import os

API_KEY = os.environ.get("CENSUS_API_KEY", "YOUR_KEY_HERE")

variables = [
    "B01003_001E",  # total pop
    "B19013_001E",  # median hh income
    "B25077_001E",  # median home value
    "B25003_001E",  # total housing tenure
    "B25003_002E",  # owner occupied
    "B25003_003E",  # renter occupied
    "B25002_001E",  # total occupancy
    "B25002_002E",  # occupied
    "B25002_003E",  # vacant
    "B01002_001E",  # median age
    "B25034_001E",  # year built total
    "B25034_010E",  # built 1939 or earlier
    "B25024_002E",  # 1-unit detached
    "B25024_003E",  # 1-unit attached
    "B02001_001E",  # race total
    "B02001_002E",  # white alone
    "B02001_003E",  # black alone
    "B03001_001E",  # hispanic origin total
    "B03001_003E",  # hispanic
]

var_string = ",".join(variables)
years = list(range(2011, 2024))

all_dfs = []
for year in years:
    print(f"Pulling ACS {year} (ZCTA)...")
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{var_string}",
        "for": "zip code tabulation area:*",
        "key": API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["acs_year"] = year
        all_dfs.append(df)
        print(f"  {year}: {len(df)} ZCTAs")
    except Exception as e:
        print(f"  {year}: FAILED — {e}")
    time.sleep(1)

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_parquet("data/raw/census/acs_zcta_demographics.parquet", index=False)
print(f"\nTotal: {len(combined)} ZCTA-year observations")
EOF
```

#### 3B. ZCTA-to-County Crosswalk

```
Source: https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html
```

**Purpose:** Assign each ZCTA to its primary county for merging with county-level FEMA declarations and the running variable. Only affects treatment assignment, not outcome measurement.

```bash
python3 << 'EOF'
import requests
import pandas as pd

url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt"
print(f"Downloading ZCTA-County relationship file...")

try:
    df = pd.read_csv(url, sep="|", dtype=str)
    df.to_csv("data/raw/census/zcta_county_crosswalk.csv", index=False)
    print(f"Downloaded: {len(df)} ZCTA-county pairs")
except Exception as e:
    print(f"Download failed: {e}")
    print("Manually download from Census relationship files page.")
EOF
```

### Phase 4: State Fiscal Capacity and Threshold Data

#### 4A. Treasury Total Taxable Resources (TTR)

```
Source: https://home.treasury.gov/policy-issues/economic-policy/total-taxable-resources
```

**Purpose:** FEMA's IA recommendation algorithm weights a state's Total Taxable Resources when evaluating declarations. High-TTR states face an effectively higher damage threshold. The TTR index is needed to construct the adjusted running variable.

**TTR is published by the U.S. Treasury, Office of Economic Policy.** It estimates each state's total tax base (income, property, consumption) and is used by multiple federal programs as a measure of state fiscal capacity.

```bash
mkdir -p data/raw/treasury

python3 << 'EOF'
"""
Download Treasury Total Taxable Resources (TTR) data.

TTR is published as Excel files on Treasury's website. The data includes
state-level TTR estimates and per-capita TTR for each fiscal year.

NOTE: The exact URL and file format may change. The TTR page is at:
https://home.treasury.gov/policy-issues/economic-policy/total-taxable-resources

As of 2025, TTR data is available in Excel format going back to FY 1981.
The file typically contains:
  - State name
  - Total Taxable Resources (millions $)
  - Per-capita TTR
  - TTR as % of US average

If the direct download fails, manually download from the Treasury page above.
"""
import requests
import pandas as pd

# Treasury TTR page — the Excel file URL changes with updates
# Try the most recent known URL pattern
ttr_urls = [
    "https://home.treasury.gov/system/files/131/TTR-Table1.xlsx",
    "https://home.treasury.gov/system/files/131/Total-Taxable-Resources.xlsx",
]

downloaded = False
for url in ttr_urls:
    try:
        print(f"Trying: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        outpath = "data/raw/treasury/ttr_raw.xlsx"
        with open(outpath, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded TTR data: {len(resp.content)} bytes")
        downloaded = True

        # Attempt to parse — TTR files have complex multi-header layouts
        # This will likely need manual inspection and cleanup
        try:
            df = pd.read_excel(outpath, sheet_name=0, header=None)
            df.to_csv("data/raw/treasury/ttr_raw_parsed.csv", index=False)
            print(f"Parsed: {df.shape[0]} rows × {df.shape[1]} cols")
            print("NOTE: TTR Excel files have complex layouts. Manual inspection required.")
            print("Key columns needed: state, fiscal_year, ttr_total, ttr_per_capita")
        except Exception as e:
            print(f"Parse warning: {e}")
            print("Excel downloaded but parsing needs manual cleanup.")
        break
    except Exception as e:
        print(f"  Failed: {e}")

if not downloaded:
    print("\nCould not auto-download TTR data.")
    print("Manual download required from:")
    print("  https://home.treasury.gov/policy-issues/economic-policy/total-taxable-resources")
    print("Save to: data/raw/treasury/ttr_raw.xlsx")
    print("\nAfter download, parse into CSV with columns:")
    print("  state_fips, fiscal_year, ttr_total_millions, ttr_per_capita, ttr_pct_us_avg")
    print("Save to: data/raw/treasury/ttr_clean.csv")
EOF
```

**TTR Adjustment Construction:**

```python
# preprocessing/04b_construct_ttr_adjustment.py
"""
Construct the TTR-adjusted running variable.

The exact functional form FEMA uses to weight TTR is not publicly documented.
We test three specifications:

1. Linear: AdjRunVar = (PerCapDamage / TTR_pct_us_avg) - Threshold
   Interpretation: $1 of per-capita damage matters more in a low-TTR state

2. Log: AdjRunVar = (PerCapDamage * log(100/TTR_pct_us_avg)) - Threshold
   Allows diminishing returns of fiscal capacity

3. Rank: AdjRunVar = (PerCapDamage * TTR_rank_weight) - Threshold
   Where TTR_rank_weight = 1 + (1 - TTR_percentile_rank)
   Bottom-quartile states get ~1.75x weight, top-quartile ~1.25x

Report all main results under all three specifications.
The specification that produces the strongest first stage is the preferred
instrument, with the others as robustness.
"""
```

#### 4B. Historical FEMA IA Threshold Values

```bash
python3 << 'EOF'
import pandas as pd

# FEMA per-capita thresholds for Individual Assistance (44 CFR 206.48)
# Updated annually in Federal Register. Values approximate — VERIFY ALL.
# The county threshold was restructured in 2008 and again in 2016.

thresholds = [
    {"year": 2000, "state_threshold": 1.07, "county_threshold": 3.20},
    {"year": 2001, "state_threshold": 1.09, "county_threshold": 3.27},
    {"year": 2002, "state_threshold": 1.11, "county_threshold": 3.33},
    {"year": 2003, "state_threshold": 1.12, "county_threshold": 3.36},
    {"year": 2004, "state_threshold": 1.14, "county_threshold": 3.42},
    {"year": 2005, "state_threshold": 1.16, "county_threshold": 3.48},
    {"year": 2006, "state_threshold": 1.19, "county_threshold": 3.57},
    {"year": 2007, "state_threshold": 1.22, "county_threshold": 3.66},
    {"year": 2008, "state_threshold": 1.29, "county_threshold": 3.50},
    {"year": 2009, "state_threshold": 1.31, "county_threshold": 3.56},
    {"year": 2010, "state_threshold": 1.35, "county_threshold": 3.61},
    {"year": 2011, "state_threshold": 1.37, "county_threshold": 3.64},
    {"year": 2012, "state_threshold": 1.39, "county_threshold": 3.68},
    {"year": 2013, "state_threshold": 1.41, "county_threshold": 3.72},
    {"year": 2014, "state_threshold": 1.43, "county_threshold": 3.78},
    {"year": 2015, "state_threshold": 1.46, "county_threshold": 3.86},
    {"year": 2016, "state_threshold": 1.47, "county_threshold": 3.50},
    {"year": 2017, "state_threshold": 1.50, "county_threshold": 3.57},
    {"year": 2018, "state_threshold": 1.52, "county_threshold": 3.62},
    {"year": 2019, "state_threshold": 1.55, "county_threshold": 3.68},
    {"year": 2020, "state_threshold": 1.57, "county_threshold": 3.72},
    {"year": 2021, "state_threshold": 1.59, "county_threshold": 3.78},
    {"year": 2022, "state_threshold": 1.63, "county_threshold": 3.86},
    {"year": 2023, "state_threshold": 1.68, "county_threshold": 3.97},
    {"year": 2024, "state_threshold": 1.74, "county_threshold": 4.12},
    {"year": 2025, "state_threshold": 1.79, "county_threshold": 4.24},
]

# VERIFY ALL VALUES against Federal Register notices before analysis.
# Confidence: 6/10 for exact figures.

df = pd.DataFrame(thresholds)
df.to_csv("data/raw/fema/ia_thresholds.csv", index=False)
print(df.to_string(index=False))
EOF
```

---

## Preprocessing Pipeline

### Step 1: Standardize Geographies

```python
# preprocessing/01_standardize_geography.py
"""
Create consistent geography linkages across all datasets.

Tasks:
1. Standardize state + county FIPS to 5-digit codes across FEMA, NOAA, Census
2. Assign each ZCTA to its primary county using Census relationship file
   (largest population share). This determines which county-level disaster
   declaration applies to each ZIP.
3. Handle FIPS code changes over time (county boundary changes, FIPS deprecations)
4. Validate FEMA damagedZipCode values against ZCTA universe; flag orphan ZIPs
   (PO Box-only, military, unique-entity) for exclusion
5. Output: ZIP-level panel dataset skeleton with consistent geographic IDs
   and county assignments
"""
```

### Step 2: Construct Event-Level Dataset

```python
# preprocessing/02_construct_events.py
"""
Build the event-level analysis dataset.

Tasks:
1. Match FEMA disaster declarations to NOAA storm events by date + geography
2. For each event, compute:
   a. Total estimated damage (PA projects + IA claims) by county
   b. Per-capita damage using Census population
   c. TTR-adjusted running variable = f(per_capita_damage, TTR) - threshold[year]
      (compute under all three TTR specifications: linear, log, rank)
   d. IA declaration status (treatment indicator)
3. Identify "near-threshold" events (|running_var| < k for various bandwidths k)
4. Flag events with potential threshold manipulation (for McCrary test)
5. Build historical IA exposure panel: for each ZIP × event, identify the most
   recent prior disaster event in the same county and whether IA was received.
   Record: prior_ia_received, prior_running_var, years_since_prior_event,
   prior_hazard_type, prior_ia_amount.
   Bin years_since_prior into: 0-2, 3-5, 6-9, 10+ years (Gallagher 2014
   found ~9-year decay in insurance take-up; calibrate temporal restrictions
   against this benchmark). Restrict Mechanism A primary sample to prior
   events within 10 years.
6. Political cycle variables (Schneider and Kunze 2025):
   a. presidential_election_year = 1 if event year % 4 == 0
   b. governor_president_alignment = 1 if same party
   c. county_presidential_vote_margin (from most recent election)
   d. Merge presidential party from election data; merge governor party
      from state government records
7. Hazard type flags: tag each event as flood-only, hurricane, severe storm,
   or mixed. The flood-only subsample is a key robustness sample because
   political bias in declarations is concentrated in hurricane events.
8. Output: event-county panel with running variables, treatment status,
   historical IA exposure linkages, election cycle indicators, and
   hazard type classifications
"""
```

### Step 3: Construct ZIP-Level Outcomes

```python
# preprocessing/03_construct_zip_outcomes.py
"""
Build ZIP-level outcome variables for both mechanisms.

Tasks:
1. Aggregate NFIP claims to ZIP level (native — no crosswalk needed)
2. Aggregate FEMA IA claims to ZIP level (native via damagedZipCode)
3. Compute Mechanism A outcomes (pre-disaster mitigation proxies):
   a. contents_damage_ratio = contents_paid / (contents_paid + building_paid)
   b. damage_per_foot_of_flood = total_paid / avg_water_depth
   c. total claim severity relative to hazard intensity controls
4. Compute Mechanism B outcomes (post-disaster claiming behavior):
   a. ia_application_rate = n_ia_applications / n_housing_units
   b. ia_approval_rate = n_approved / n_applications
   c. avg_ia_award = total_approved / n_approved
   d. time_to_ia_application (if timestamps available)
   e. sba_loan_application_rate (if SBA data merged)
5. Compute tenure interaction variables:
   a. ZCTA homeownership rate from ACS B25003
   b. Split owner vs. renter IA amounts using HousingAssistanceOwners/Renters
6. Merge ZCTA demographics from ACS (direct join on 5-digit ZIP/ZCTA code)
7. Output: ZIP-event panel with outcomes and controls
"""
```

### Step 4: Construct Hazard Intensity and Warning Lead Time Controls

```python
# preprocessing/04_hazard_intensity.py
"""
Build ZIP-level hazard exposure and warning lead time measures.

Tasks:
1. Compute ZIP centroid coordinates from ZCTA shapefiles (Census TIGER)
2. For flood events:
   a. Interpolate USGS high water marks to ZIP centroids (IDW or kriging)
   b. Match NOAA storm event damage estimates by county
3. For hurricane events:
   a. Compute max sustained wind speed at ZIP centroid from HURDAT2 track data
      using a parametric wind field model (Holland 1980 or similar)
   b. Compute rainfall accumulation from NOAA precipitation data (if available)
4. For severe storms:
   a. Tornado path width × F-scale at ZIP centroid
   b. Hail diameter reports near ZIP centroid
5. WARNING LEAD TIME (critical for Mechanism A):
   a. For each event × county, find earliest NWS warning issued
      (PHENOM in [FF, FL, HU, TO, SV], SIG=W)
   b. Compute lead_time_hours = incidentBeginDate - earliest_warning_issued
   c. Classify: flash (<6h), short_notice (6-24h), adequate_notice (≥24h)
   d. For Mechanism A primary sample: restrict to adequate_notice events
   e. Robustness: test across 6h, 12h, 24h, 48h warning thresholds
6. Output: ZIP-event hazard intensity + warning lead time measures
"""
```

### Step 5: Build Analysis Datasets

```python
# preprocessing/05_build_analysis_dataset.py
"""
Merge all components into final analysis-ready datasets.

Tasks:
1. Merge: events × ZIPs × outcomes × hazard_intensity × demographics × TTR
   × election_cycle_indicators
2. Build THREE analysis datasets:

   DATASET A (Mechanism A — Historical IV, primary):
   a. Restrict to ZIPs with a prior disaster event in the same county
   b. Require prior event running variable within bandwidth of threshold
   c. Require prior event within 10 years (Gallagher 2014 temporal decay)
   d. Require current event warning lead time ≥24h (primary); test robustness
   e. Include: prior_ia_received, prior_running_var, years_since_prior_event,
      years_since_prior_bin (0-2, 3-5, 6-9, 10+), current_mitigation_outcomes,
      current_hazard_intensity, ZCTA demographics, homeownership_rate,
      warning_hours, election_cycle_vars
   f. Drop ZIPs with zero population or missing ZCTA demographics

   DATASET A_FLOOD (Mechanism A — flood-only robustness):
   a. Same as Dataset A but restricted to events where incidentType is
      Flood or Coastal (excluding Hurricane, Severe Storm, Tornado)
   b. This sample is less susceptible to political bias in declarations
      (Schneider and Kunze 2025 finding is hurricane-specific)
   c. Also has better hazard intensity controls (USGS high water marks)

   DATASET B (Mechanism B — Contemporaneous RD):
   a. Current events within bandwidth of threshold (±0.25, ±0.50, ±1.00 per capita)
   b. ZIPs with >0 NFIP policies or >0 IA applications
   c. Include: ia_declaration_status, current_running_var, post_disaster_outcomes,
      ZCTA demographics, homeownership_rate, hazard_intensity, election_cycle_vars
   d. Drop ZIPs with zero population or missing ZCTA demographics
   e. Also produce flood-only variant (dataset_b_flood.parquet)

3. Compute SDID matching variables for all datasets:
   a. Pre-event demographic similarity
   b. Historical hazard exposure
   c. Prior FEMA declaration history
   d. Baseline NFIP take-up rate
4. Output: analysis parquet files (mechanism_a.parquet, mechanism_a_flood.parquet,
   mechanism_b.parquet, mechanism_b_flood.parquet)
"""
```

---

## Analysis Plan

### Stage 1: Descriptive Statistics and Validation

1. **Summary statistics** by declaration status (IA declared vs. not) for all datasets
2. **McCrary density test** on the running variable — test for manipulation at threshold (both historical and contemporaneous)
3. **Balance tests** at the threshold — demographics, housing stock, hazard exposure should be smooth through the cutoff
4. **Placebo RDs on predetermined covariates:** Test for discontinuities in median household income and county-level presidential vote margin at the threshold. If these show significant jumps, the threshold is capturing political sorting rather than quasi-random variation.
5. **Election cycle first-stage diagnostic (Schneider and Kunze 2025):**
   - Split sample by presidential election year vs. non-election year
   - Split by governor-president party alignment
   - If the first stage (threshold → declaration) is substantially stronger in election years or aligned-party states, political sorting is contaminating the threshold. Report separately and flag.
6. **First stage diagnostics:**
   - Mechanism A: Plot historical IA receipt probability against the historical running variable; report first-stage F-statistic
   - Mechanism A: Plot first-stage F by years-since-prior-event bin (0-2, 3-5, 6-9, 10+) to identify where the instrument has power. Compare temporal decay to Gallagher (2014) benchmark (~9-year half-life for insurance take-up).
   - Mechanism B: Plot current IA declaration probability against the contemporaneous running variable; report first-stage F-statistic
   - Test first-stage strength under all three TTR adjustment specifications; select preferred instrument
   - Report all first-stage diagnostics separately for flood-only and all-hazard samples

### Stage 2: Mechanism A — Historical IV Estimates (Pre-Disaster Mitigation)

**This is a two-sided test.** β₁ < 0 supports the Peltzman/substitution hypothesis; β₁ > 0 supports the Andor/complementarity hypothesis. Report two-sided p-values throughout. Do not use directional language in results descriptions.

1. **First stage:** Regress prior IA receipt on historical threshold crossing, controlling for the running variable polynomial
2. **Second stage:** Estimate effect of predicted prior IA receipt on current-event mitigation outcomes:
   - Contents damage ratio (primary)
   - Damage per foot of flood depth
   - Total claim severity relative to hazard intensity
3. **Warning lead time restriction:** Primary estimates on ≥24h warning sample; robustness across 6h, 12h, 48h thresholds
4. **Tenure heterogeneity:** Interact fitted IA receipt with ZCTA homeownership rate. Test whether effect differs for high- vs. low-homeownership ZIPs. Andor et al. (2020) found that the complementarity between aid and physical mitigation was strongest for non-financial measures; Botzen et al. (2019) found substitution for emergency measures but complementarity for structural measures. This project tests whether the U.S. institutional context produces the same pattern or a different one.
5. **Temporal decay (calibrated against Gallagher 2014):**
   - Estimate separate effects by years-since-prior bin (0-2, 3-5, 6-9, 10+)
   - Plot β₁ by bin to map the decay curve
   - Compare to Gallagher's ~9-year half-life for insurance take-up
   - If the effect decays at a similar rate, the expectations channel is the likely mechanism
   - If the effect does not decay (or strengthens), a non-expectations channel (infrastructure, building codes) may dominate, weakening the IV exclusion restriction
   - Primary specification restricts to prior events within 10 years; report 5-year and 15-year robustness
6. **Bandwidth selection:** Imbens-Kalyanaraman optimal bandwidth + robustness across alternatives
7. **Controls:** Demographics, hazard intensity, year FE, state FE, hazard-type FE, election cycle indicators

### Stage 3: Mechanism B — Contemporaneous RD Estimates (Post-Disaster Claiming)

1. **Local linear regression** with triangular kernel around the contemporaneous threshold
2. **Outcomes:**
   - IA application rate (primary)
   - IA approval rate and average award
   - NFIP claim severity relative to hazard intensity
   - SBA disaster loan application rate
   - Time to claim filing
3. **Bandwidth selection:** Imbens-Kalyanaraman optimal bandwidth + robustness across alternatives
4. **Specifications:**
   - Sharp RD (threshold → outcome directly)
   - Fuzzy RD (threshold → IA declaration → outcome, using threshold as IV)
   - With and without controls (demographics, hazard intensity, year FE, state FE)
5. **Tenure heterogeneity:** Split by ZCTA homeownership rate. Owners eligible for full IHP housing assistance; renters only for ONA.

### Stage 4: Synthetic Difference-in-Differences (SDID)

Replaces the original SCM approach. Classical Abadie-style SCM is designed for comparative case studies with few treated units and is computationally intractable and prone to overfitting at the ZIP level with thousands of treated units.

**Method:** Synthetic Difference-in-Differences (Arkhangelsky, Athey, Hirshberg, Imbens, Peel, 2021). SDID combines the unit-weighting intuition of synthetic controls (reweight control units to match treated-unit pre-treatment outcomes) with the time-weighting of difference-in-differences (reweight pre-treatment periods to match post-treatment periods). This produces a doubly-robust estimator that is valid under weaker assumptions than either SCM or DiD alone.

**Implementation:**
1. Define treatment as IA declaration at the county level, applied to all ZIPs in the declared county
2. Pre-treatment period: prior disaster events and inter-event periods for the same ZIP
3. Post-treatment period: the current disaster event
4. Unit weights: constructed to match pre-treatment outcome trajectories of treated ZIPs
5. Time weights: constructed to match the pre-treatment pattern most predictive of post-treatment outcomes
6. Inference: Placebo-based — randomly reassign treatment across counties, construct distribution of placebo SDID estimates

**Apply to both mechanisms:**
- Mechanism A: Treatment = prior IA receipt; outcome = current mitigation behavior
- Mechanism B: Treatment = current IA declaration; outcome = post-disaster claiming

**Software:** Use the `synthdid` R package (Arkhangelsky et al.) via `rpy2`, or the Python implementation `pysynthdid` if available and stable.

### Stage 5: Robustness and Heterogeneity

1. **Placebo thresholds** — test at fake thresholds where no discontinuity should exist
2. **Donut RD** — exclude observations very close to threshold (manipulation concern)
3. **TTR specification robustness** — report all main results under linear, log, and rank TTR adjustments
4. **Flood-only robustness (Schneider and Kunze 2025):**
   - Re-estimate all main specifications on flood-only subsamples (mechanism_a_flood.parquet, mechanism_b_flood.parquet)
   - Schneider and Kunze found political bias in declarations is concentrated in hurricane events; floods are less politically salient
   - If main results hold in flood-only sample, political contamination is less likely to explain findings
   - Flood events also have better hazard intensity controls (USGS high water marks vs. wind field models)
5. **Election cycle heterogeneity (Schneider and Kunze 2025):**
   - Interact treatment with presidential election year indicator
   - Interact treatment with governor-president alignment indicator
   - If effects are concentrated in election years or aligned-party states, political sorting rather than the damage threshold may be driving declarations
   - Report non-election-year subsample as additional robustness
6. **Heterogeneity by:**
   - Hazard type (flood vs. hurricane vs. severe storm)
   - Income level (low-income ZIPs may face different tradeoffs)
   - Prior disaster experience (repeat vs. first-time IA areas)
   - Homeownership rate (owners vs. renters face different mitigation incentives; Andor et al. 2020 and Botzen et al. 2019 suggest different effects by mitigation type)
   - Warning lead time (gradient of effect with advance notice hours)
   - Time period (pre/post FEMA reform years — 2008, 2016 restructurings)
7. **Mechanism interaction tests:**
   - Does prior IA experience (Mechanism A) amplify the post-disaster claiming response (Mechanism B)?
   - Does IA availability affect NFIP take-up? (charity hazard in insurance; Kousky et al. 2018 found substitution here)
   - Does IA availability affect SBA disaster loan applications? (substitution)
   - Does the effect vary by IA grant size? (dose-response)
8. **Scope test (Deryugina 2017):**
   - The IA declaration signals broader federal support beyond the grant itself
   - Test whether Mechanism B effects correlate with total post-disaster transfer increases (UI claims, Medicaid enrollment) in the same county-year
   - If they do, the treatment is best interpreted as "declared disaster area status" not "IA grant receipt"

---

## Project Directory Structure

```
fema-peltzman/
├── README.md                          # This file
├── pyproject.toml                     # Dependencies
├── .env                               # API keys (CENSUS_API_KEY)
├── data/
│   ├── raw/
│   │   ├── fema/                      # OpenFEMA downloads
│   │   ├── noaa/                      # Storm events
│   │   ├── nws/                       # NWS warning archives (IEM)
│   │   ├── usgs/                      # Flood data
│   │   ├── nhc/                       # Hurricane tracks
│   │   ├── treasury/                  # TTR data
│   │   ├── elections/                 # Presidential/governor party data
│   │   └── census/                    # ACS ZCTA data + crosswalks
│   ├── intermediate/                  # Cleaned/standardized
│   └── analysis/                      # Final analysis datasets
│       ├── mechanism_a.parquet        # Historical IV dataset (all hazards)
│       ├── mechanism_a_flood.parquet  # Historical IV dataset (flood-only)
│       ├── mechanism_b.parquet        # Contemporaneous RD dataset (all hazards)
│       └── mechanism_b_flood.parquet  # Contemporaneous RD dataset (flood-only)
├── preprocessing/
│   ├── 01_standardize_geography.py
│   ├── 02_construct_events.py         # Includes election cycle + temporal bins
│   ├── 03_construct_zip_outcomes.py
│   ├── 04_hazard_intensity.py         # Includes warning lead time
│   ├── 04b_construct_ttr_adjustment.py
│   └── 05_build_analysis_dataset.py   # Produces all four analysis datasets
├── analysis/
│   ├── 01_descriptive_stats.py
│   ├── 02_mccrary_test.py
│   ├── 03_balance_tests.py
│   ├── 04_placebo_rd_covariates.py    # Income + vote margin placebos
│   ├── 05_first_stage.py              # Both mechanisms + election cycle split
│   ├── 06_mechanism_a_historical_iv.py # Two-sided test
│   ├── 07_mechanism_b_contemp_rd.py
│   ├── 08_sdid_estimates.py
│   ├── 09_robustness.py               # Includes flood-only + election cycle
│   └── 10_heterogeneity.py            # Includes tenure × Andor/Botzen framework
├── figures/
├── tables/
└── paper/
    ├── draft.md
    └── literature_review.md
```

## Dependencies

```toml
[project]
name = "fema-peltzman"
requires-python = ">=3.10"

dependencies = [
    "pandas>=2.0",
    "pyarrow",
    "requests",
    "numpy",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "rdrobust",           # RD estimation (Calonico, Cattaneo, Titiunik)
    "rdd",                # Additional RD tools
    "linearmodels",       # Panel estimators, IV
    "matplotlib",
    "seaborn",
    "geopandas",
    "shapely",
    "beautifulsoup4",
    "tqdm",
    "openpyxl",           # For parsing Treasury TTR Excel files
    "rpy2",               # R interface for synthdid package
    "pysynthdid",         # Python SDID implementation (if stable; fallback to rpy2)
]
```

**R dependency (for SDID):**

```bash
# Install synthdid R package
Rscript -e 'install.packages("synthdid", repos="https://cran.r-project.org")'
# Also install did package for Callaway & Sant'Anna as additional robustness
Rscript -e 'install.packages("did", repos="https://cran.r-project.org")'
```

---

## Known Risks and Limitations

1. **Historical IV relevance.** The first stage for Mechanism A depends on households generalizing from prior IA experience to current expectations. If the prior event was many years ago, a different hazard type, or in a different political context, the IV may be weak. Gallagher (2014) found ~9-year decay in insurance take-up, suggesting the instrument is relevant primarily for prior events within the last decade. Test first-stage F by years-since-prior bin and by hazard-type match. Confidence in adequate first stage: 5/10.

2. **Historical IV exclusion restriction.** Prior IA receipt may affect current outcomes through channels other than expectations — rebuilt infrastructure, changed drainage, updated building codes triggered by the prior disaster. These mechanical channels are likely to produce *less* damage (lower contents ratio, lower claim severity), regardless of whether the behavioral channel produces more or less mitigation. If mechanical channels dominate, the IV captures infrastructure effects, not expectations. The temporal decay test (point 5 in Stage 2) helps discriminate: expectation-driven effects should decay over ~9 years (Gallagher 2014), while infrastructure effects persist indefinitely.

3. **Complementarity vs. substitution is genuinely ambiguous.** Andor et al. (2020) and Botzen et al. (2019) found that physical mitigation and disaster aid expectations are complements in German and Sandy-affected households. The U.S. federal system — with its presidential declarations, media spectacle, and explicit IA grants — may produce different behavioral responses. But the project cannot assume the Peltzman direction ex ante. This is a feature, not a bug: the two-sided test is more informative than a directional test would be. Confidence that the sign will be interpretable given the proxy outcomes: 5/10.

4. **TTR adjustment specification uncertainty.** FEMA's exact TTR weighting formula is not publicly documented. The three specifications (linear, log, rank) are reasonable guesses. If none produce a strong first stage, the running variable may be fundamentally misspecified. Confidence in getting the adjustment right: 5/10. Early diagnostic priority.

5. **Political cycle contamination (Schneider and Kunze 2025).** Political bias in disaster declarations concentrates in medium-intensity events — exactly where the per-capita damage threshold operates. If the first stage is substantially stronger in election years or aligned-party states, the threshold is capturing political sorting rather than quasi-random hazard variation. The flood-only robustness sample partially addresses this (the Schneider-Kunze finding is hurricane-specific), but floods are not immune to political influence. Confidence that political contamination is manageable with the planned diagnostics: 6/10.

6. **NWS warning data coverage.** IEM archives are comprehensive but may have gaps in early years (pre-2005). Warning lead time computation depends on matching warning polygons to county geographies, which requires spatial joins. Data quality degrades before 2000. Confidence in clean warning lead time: 6/10.

7. **ZCTA-to-county assignment for treatment status.** ZCTAs spanning county boundaries get assigned to one county. Misclassification at the boundary attenuates the first stage but doesn't bias the treatment effect estimate. Confidence: 8/10.

8. **Per-capita threshold is necessary but not sufficient.** FEMA considers qualitative factors beyond the per-capita threshold. The fuzzy RD accounts for this, but both first stages may be weaker than expected. Confidence in adequate first-stage F for Mechanism B: 6/10.

9. **Behavioral outcomes are indirect.** We don't observe mitigation behavior directly — we infer it from damage patterns. The contents-to-structural damage ratio is a reasonable proxy, but other unobserved factors affect it. Confidence in outcome validity: 5/10.

10. **Threshold values need verification.** The IA per-capita thresholds listed are approximate. Federal Register cross-verification required. Restructurings in 2008 and 2016 may introduce structural breaks. Confidence: 6/10.

11. **Sample size for Mechanism A.** Restricting to ZIPs with (a) prior near-threshold events within 10 years, (b) current events with ≥24h warning, and (c) adequate NFIP/IA claims may produce a small effective sample. The flood-only subsample will be smaller still. Power analysis is critical before committing to specifications.

12. **ZIP geographic coarseness.** ZIPs are larger than census tracts. Within-ZIP heterogeneity is averaged away. Acceptable tradeoff given measurement error gains from avoiding crosswalks.

13. **Media interaction is speculative.** The media intensity index requires GDELT or similar data and is hard to construct cleanly at the county × pre-event-window level. Treat as an extension, not core. Confidence: 4/10.

14. **IA declaration signals broader federal support (Deryugina 2017).** Non-disaster transfers (UI, Medicaid, income maintenance) after hurricanes significantly exceed direct disaster aid. The Mechanism B treatment is best understood as "household learns it is in a declared disaster area" — capturing the full signaling value of the declaration, not just the IA grant. This is arguably the more policy-relevant quantity, but means the estimated effect cannot be attributed to IA specifically. The scope test in Stage 5 (correlation with total transfer increases) helps assess how much of the effect operates through IA versus broader safety net expectations.

---

## Execution Order for Claude Code Agent

```
STEP 1:  Create project directory structure
STEP 2:  Set up Python environment (pip install dependencies)
STEP 3:  Install R and synthdid package
STEP 4:  Download Phase 1 data (FEMA — largest, start first)
STEP 5:  Download Phase 2 data (NOAA storm events, USGS, NHC — can parallelize)
STEP 6:  Download Phase 2D data (NWS warnings from IEM — new, may be slow)
STEP 7:  Download Phase 3 data (Census ACS ZCTA — requires API key)
STEP 8:  Download Phase 4 data (Treasury TTR — may need manual download)
STEP 9:  Download election cycle data (presidential party by year, governor
         party by state-year, county-level presidential vote margins)
         Sources: MIT Election Data + Science Lab, Carl Klarner state
         politics dataset, or Dave Leip's Atlas
STEP 10: Run preprocessing Step 1 (geography standardization)
STEP 11: Run preprocessing Step 2 (event construction + historical linkages
         + election cycle indicators + temporal decay bins)
STEP 12: Run preprocessing Step 4b (TTR adjustment — test three specifications)
STEP 13: Run preprocessing Step 3 (ZIP-level outcomes)
STEP 14: Run preprocessing Step 4 (hazard intensity + warning lead time)
STEP 15: Run preprocessing Step 5 (build all four analysis datasets:
         mechanism_a, mechanism_a_flood, mechanism_b, mechanism_b_flood)
STEP 16: Run Stage 1 analysis (descriptive + validation + placebo RDs
         + election cycle first-stage diagnostic)
STEP 17: Evaluate first-stage diagnostics for both mechanisms — STOP AND ASSESS
         before proceeding. If first stages are weak, revisit TTR specification
         or bandwidth choices. If election cycle diagnostic fails, consider
         restricting to flood-only or non-election years.
STEP 18: Run Stages 2-5 analysis (two-sided tests for Mechanism A)
```

**Estimated total data volume:** ~8-20 GB raw (NWS warnings are large), ~2-4 GB processed.

**Critical dependencies:**
- Census API key: Register at https://api.census.gov/data/key_signup.html before Step 7
- R installation with synthdid package for Step 18 (SDID analysis)
- Treasury TTR data may require manual download if automated URLs fail (Step 8)
- Election data (Step 9): MIT Election Data + Science Lab (https://electionlab.mit.edu/data) for county-level presidential vote; state governor party from Klarner dataset or similar

**Rate limiting:** OpenFEMA API — 0.3-0.5s delays. Census API — 500/day without key, unlimited with key. IEM — be polite, 2s delays between year downloads.

**STOP-AND-ASSESS checkpoint at Step 16:** Do not proceed to main estimation if:
- McCrary test rejects (threshold manipulation)
- Placebo RDs on predetermined covariates show significant discontinuities
- First-stage F < 10 under all TTR specifications
- Mechanism A sample size < 500 ZIP-events after all restrictions (including 10-year temporal restriction)
- First-stage F is >2x stronger in presidential election years than non-election years (political sorting dominates)
If any of these fail, return to this document and reassess design choices before burning compute on main estimates.

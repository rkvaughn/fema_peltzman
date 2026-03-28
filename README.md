# FEMA Peltzman

Does exposure to FEMA Individual Assistance alter household pre-disaster mitigation effort and post-disaster claiming behavior?

This project tests the "Peltzman hypothesis" — that federal disaster aid substitutes for private mitigation — against the competing complementarity hypothesis (Andor et al. 2020; Botzen et al. 2019), using two quasi-experimental designs:

- **Mechanism A:** Historical IV — prior IA receipt (instrumented by past threshold crossing) → pre-disaster mitigation proxies from NFIP damage patterns
- **Mechanism B:** Contemporaneous fuzzy RD — current IA declaration threshold → post-disaster claiming behavior

## Setup

```bash
# Requires R with synthdid, did, rddensity packages installed first
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # add CENSUS_API_KEY
```

## Pipeline

```
acquire/01_fema.py               # OpenFEMA: declarations, IA, NFIP, PA
acquire/02_noaa_storm_events.py  # NOAA NCEI storm events 2000–2024
acquire/03_hazard_point_sources.py  # USGS STN, HURDAT2, NWS warnings (IEM)
acquire/04_census.py             # ACS ZCTA demographics + TIGER shapefiles
acquire/05_ancillary.py          # Treasury TTR, election data, threshold stub

preprocessing/01_standardize_geography.py
preprocessing/02_construct_events.py
preprocessing/03_construct_zip_outcomes.py
preprocessing/04_hazard_intensity.py
preprocessing/04b_construct_ttr_adjustment.py  # ← threshold verification gate
preprocessing/05_build_analysis_dataset.py

analysis/00_gate_check.py        # ← McCrary / balance / F-stat / sample size gates
analysis/01_descriptive_stats.py
analysis/02_mccrary_test.py
analysis/03_balance_tests.py
analysis/04_placebo_rd_covariates.py
analysis/05_first_stage.py
analysis/06_mechanism_a_historical_iv.py
analysis/07_mechanism_b_contemp_rd.py
analysis/08_sdid_estimates.py
analysis/09_robustness.py
analysis/10_heterogeneity.py
```

## Data integrity

Per project protocol, no quantitative values are hardcoded. The IA threshold table (`data/raw/fema/ia_thresholds_UNVERIFIED.csv`) must be verified against the Federal Register before `04b` will run. See `CLAUDE.md` for full data integrity rules.

## Dependencies

Python: pandas, geopandas, numpy, scipy, statsmodels, linearmodels, rdrobust, rpy2, scikit-learn, matplotlib, seaborn, requests

R (via rpy2): synthdid, did, rddensity

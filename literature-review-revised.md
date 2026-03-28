# Literature Review: Moral Hazard and Emergency Disaster Aid (Revised)

## Summary of Revisions from Original Draft

**Citation corrections:** Two citations in the original draft were misattributed. Andor, Frondel, and Sommer (2017) should be Andor, Osberghaus, and Simora (2020). Gallego et al. (2023) should be Schneider and Kunze (2023/2025). Details below.

**Missing literature added:** Gallagher (2014), Deryugina (2017), Beatty, Shimshack, and Volpe (2019), and Botzen, Kunreuther, and Michel-Kerjan (2019). These have direct implications for the research design.

**Design revision proposed:** The Andor et al. and Botzen et al. findings suggest that physical mitigation and aid expectations may be *complements* rather than substitutes. The project should frame Mechanism A as a two-sided test rather than assuming the Peltzman direction.

---

## 1. The "Charity Hazard" Hypothesis and Private Mitigation

The hypothesis that government safety nets distort private risk calculus is foundational to the economics of moral hazard (Peltzman, 1975). In the disaster context, this phenomenon is called "charity hazard" — the tendency of individuals to forgo private insurance or loss-mitigation measures due to anticipated government relief. Raschky and Weck-Hannemann (2007) formalized this in disaster economics, demonstrating theoretically that post-disaster government relief substitutes for private market insurance and hinders development of natural hazard insurance markets.

However, the empirical literature on *physical* mitigation (as opposed to financial insurance) tells a more complicated story. Andor, Osberghaus, and Simora (2020) investigated charity hazard using survey data from flood-prone German households. They found a substantial charity hazard effect in the insurance market — anticipated government aid reduced insurance take-up — but a *positive* correlation between anticipated government aid and non-financial protection measures such as sandbagging and securing utilities. Their interpretation: households view insurance and government aid as substitutes, but physical mitigation and aid as complements. Households perceive they can receive aid even if they attempt to protect their property, so aid expectations do not crowd out physical effort.

This complementarity finding is reinforced by Botzen, Kunreuther, and Michel-Kerjan (2019), who studied Hurricane Sandy homeowners in New York City. They found that precautionary structural measures and insurance coverage are complements (advantageous selection), while emergency preparedness measures and insurance may be substitutes (consistent with moral hazard). The distinction between structural and emergency mitigation types matters — the behavioral response to risk is not uniform across protective action categories.

**Design implication:** These findings directly challenge the Peltzman framing of this project. If physical mitigation and expected aid are complements in the German/Sandy context, the project's Mechanism A could produce β₁ > 0 (more mitigation in areas with IA history) rather than β₁ < 0. The research design must frame this as a *two-sided test* with competing hypotheses: (1) the Peltzman/substitution hypothesis (β₁ < 0), and (2) the Andor/complementarity hypothesis (β₁ > 0). A null finding or a positive finding is informative, not a failure. The sign and context of the result will reveal whether U.S. federal disaster aid operates more like the German system (where aid and mitigation coexist) or whether the specific features of FEMA IA — media signaling, political salience, presidential declaration drama — create stronger substitution incentives than European systems.

## 2. Learning, Expectations, and Temporal Dynamics

The project's Mechanism A — using historical IA receipt as an instrument for current expectations — sits within a broader empirical literature on how disaster experience shapes household behavior over time.

Gallagher (2014) provides the most relevant evidence on the dynamics of risk updating. Using a nationwide panel of floods and flood insurance policies, he showed that NFIP insurance take-up spikes the year after a flood and then steadily declines back to baseline over approximately 9 years. Critically, he also found that residents in non-flooded communities in the same television media market increased take-up at roughly one-third the rate of flooded communities — demonstrating that media exposure, not just direct experience, drives risk updating. The data are most consistent with a Bayesian learning model that allows for forgetting or incomplete information about past events.

**Design implications:** Gallagher's findings produce three specific requirements for the Mechanism A specification. First, the temporal decay test (already in the project plan as an interaction `IA_hat × years_since_prior`) should be calibrated against the ~9-year decay found for insurance take-up. If the mitigation-expectations channel operates similarly, we would expect the IV relevance to weaken substantially for prior events older than a decade. Second, the first-stage sample should be stratified by years-since-prior-event to identify where the instrument has adequate power. Third, the media spillover finding supports the project's secondary media interaction specification — households may update expectations from observing nearby declarations, not just from direct receipt. This could expand the donor pool for Mechanism A by including ZIPs in the same media market as a declared county, not just those in the declared county itself.

## 3. Empirical Evidence on FEMA Individual Assistance

Direct empirical studies isolating the effect of FEMA IA grants on household behavior are rare, largely because of the endogeneity of disaster declarations.

Kousky, Michel-Kerjan, and Raschky (2018) provide the most comprehensive evidence. Analyzing the effect of federal disaster assistance on NFIP flood insurance purchases, they found that receipt of FEMA disaster aid grants significantly decreased subsequent demand for flood insurance — hard empirical evidence of charity hazard in the insurance channel. While they focus on post-disaster insurance take-up rather than pre-disaster physical mitigation, their finding establishes that FEMA IA definitively alters household financial behavior in the expected direction.

Deryugina (2017) demonstrates that the scope of federal disaster support extends far beyond FEMA IA grants. Analyzing U.S. hurricanes, she showed that non-disaster government transfers — unemployment insurance, public medical payments, income maintenance — increase substantially in affected counties in the decade after a hurricane. The present value of these non-disaster transfer increases significantly exceeds that of direct disaster aid. This finding has two important implications for the project. First, it suggests that IA grants are only one component of the total post-disaster safety net. Households may respond to the full expected federal transfer package, not just the IA grant specifically — meaning the Mechanism B RD captures only a partial treatment. Second, it implies that the moral hazard channel may operate through the broader expectation of being "made whole" by the federal government, which IA declarations signal even if the IA grant itself is modest.

Shabman (2012) highlights a related behavioral nuance: actual FEMA IA payouts are historically quite low and subject to strict limitations. The moral hazard may therefore be driven more by the *perception and expectation* of generous federal aid — amplified by pre-event media narratives — than by the actuarial reality of IA grants. It is the signaling of the bailout, not the bailout itself, that triggers the Peltzman calculus.

Beatty, Shimshack, and Volpe (2019) provide the most direct behavioral evidence on pre-disaster preparedness actions. Using retail scanner data, they found that sales of emergency supplies spike before and after hurricanes, with the pre-storm response concentrated in batteries, water, and non-perishable food. While they do not directly test the moral hazard channel, their data establish that household preparedness behavior is measurable and responsive to approaching hazards. Their methodology — using retail sales as a revealed-preference measure of preparedness — could serve as a model for extending this project if private retail data become available.

## 4. The Political Economy of FEMA Thresholds

Any identification strategy relying on quasi-random variation at the FEMA threshold must account for political manipulation of disaster declarations.

Garrett and Sobel (2003) provided the foundational evidence, finding that states politically important to the president have significantly higher rates of disaster declarations, and that relief packages are systematically larger for congressional districts aligned with the incumbent party.

Schneider and Kunze (2023 working paper; published 2025, Review of Economics and Statistics) refined this finding with an important nuance: political bias in disaster declarations is *hump-shaped* in disaster severity. Decisions are unbiased when disasters are either very strong (obvious need) or very weak (obvious non-qualification). Political favoritism concentrates in medium-intensity events, where areas governed by the president's co-partisans receive up to twice as many declarations. This hump-shaped bias accounts for roughly 8% of total relief spending.

**Design implication:** This finding is directly problematic for the RD design. The per-capita damage threshold that drives the project's running variable lies squarely in the medium-intensity range — exactly where Schneider and Kunze find the most political contamination. The placebo RD tests on predetermined covariates (income, presidential vote margin) already in the project plan are essential, but may not be sufficient. An additional robustness check should restrict the sample to flood events, which are less politically salient than hurricanes (the Schneider-Kunze analysis is hurricane-specific). The project should also test for differential first-stage strength across presidential election cycles — if the first stage is substantially stronger in election years, political sorting rather than the damage threshold is likely driving declarations near the cutoff.

## 5. Synthesis and Gap Identification

The existing literature confirms four things relevant to this project:

First, the anticipation of disaster aid crowds out private insurance (Kousky et al. 2018; Andor et al. 2020), but may *not* crowd out physical mitigation — and may even complement it (Andor et al. 2020; Botzen et al. 2019). The sign of the pre-disaster mitigation effect is an empirical question, not a theoretical given.

Second, FEMA's declaration thresholds are politically porous, with bias concentrated exactly in the medium-intensity range where the RD operates (Schneider and Kunze 2023/2025). This requires robust diagnostics and potentially hazard-type restrictions.

Third, household risk expectations update from disaster experience but decay over ~9 years (Gallagher 2014), and are influenced by media exposure even without direct experience. This calibrates the temporal dynamics of the historical IV.

Fourth, the total federal safety net after disasters far exceeds direct FEMA aid (Deryugina 2017), meaning the IA declaration is best understood as a *signal* of broader federal support, not the totality of the moral hazard treatment.

**The gap:** There is no large-scale, quasi-experimental evidence linking the localized expectation of FEMA IA — whether formed through historical experience or contemporaneous declaration — to household-level physical mitigation and claiming behavior, with appropriate controls for hazard intensity and warning lead time. This project fills that gap by separating the expectations channel (Mechanism A) from the contemporaneous moral hazard channel (Mechanism B), while remaining agnostic about whether the treatment effect is positive (complementarity) or negative (substitution).

---

## Proposed Design Revisions Based on Literature Review

### Revision 1: Two-Sided Framing for Mechanism A

**Current design:** Assumes β₁ < 0 (Peltzman substitution).

**Proposed revision:** Frame as a two-sided test with competing hypotheses:
- H₁ₐ (Peltzman/substitution): β₁ < 0 — IA expectations reduce physical mitigation
- H₁ᵦ (Andor/complementarity): β₁ > 0 — IA expectations increase physical mitigation because households perceive aid is available regardless of mitigation effort, so the opportunity cost of mitigation time is lower (they don't need to spend time on insurance paperwork/financial planning and can focus on physical protection)
- H₀: β₁ = 0 — no effect

The sign of the result is informative about which institutional features of the U.S. disaster aid system dominate household decision-making. The Andor et al. complementarity result is from Germany, where the institutional context differs substantially. The U.S. system — with presidential declarations, media spectacle, and explicit IA grants — may generate different behavioral responses than European systems with more automatic disaster compensation.

**Implementation:** Change the analysis plan to report two-sided p-values and avoid directional language in pre-registration. Add a subsection to the analysis that interprets positive vs. negative coefficients against the Andor et al. framework.

### Revision 2: Calibrate Temporal Decay Against Gallagher (2014)

**Current design:** Tests temporal decay of Mechanism A effect with interaction `IA_hat × years_since_prior`.

**Proposed revision:** Explicitly model the decay curve and compare to Gallagher's ~9-year half-life for insurance take-up. Implement as:
- Bin years-since-prior into 0-2, 3-5, 6-9, 10+ year bins
- Run separate first-stage and reduced-form regressions by bin
- Plot first-stage F-statistic by years-since-prior to identify where the instrument has power
- Restrict the primary Mechanism A sample to events where the prior disaster occurred within 10 years (based on Gallagher's decay estimate)

### Revision 3: Election Cycle Heterogeneity Test

**Current design:** Does not test for political cycle effects.

**Proposed revision based on Schneider and Kunze:** Add an election-cycle heterogeneity test. If the first stage is substantially stronger in presidential election years (or in years where the governor's party aligns with the president's), political sorting rather than the damage threshold may be driving declarations near the cutoff. Implementation:
- Interact running variable with election-year indicator
- Test whether the first stage (threshold → declaration) differs by political alignment
- If it does, restrict the primary sample to non-election years or include political alignment as a control

### Revision 4: Flood-Only Robustness Sample

**Current design:** Pools all hazard types.

**Proposed revision based on Schneider and Kunze:** Since political bias in declarations is concentrated in hurricane events, add a flood-only robustness specification. Floods are less politically salient, produce a cleaner first stage, and have better hazard intensity controls (USGS high water marks). If the main results hold in the flood-only sample, political contamination is less likely to explain the findings.

### Revision 5: Acknowledge Scope Limitation Re: Total Safety Net

**Current design:** Treats FEMA IA as the treatment.

**Proposed revision based on Deryugina (2017):** Add a limitations discussion noting that IA declarations signal broader federal support (UI, Medicaid, etc.) that dwarfs the IA grant itself. The treatment in Mechanism B is best understood as "household learns it is in a declared disaster area" rather than "household receives $X in IA." This means the estimated effect captures the full signaling value of the declaration, not just the IA grant. This is arguably the more policy-relevant quantity anyway, but the distinction matters for interpretation.

---

## References (Corrected and Expanded)

* Andor, M. A., Osberghaus, D., & Simora, M. (2020). Natural Disasters and Governmental Aid: Is There a Charity Hazard? *Ecological Economics*, 169, 106534. https://doi.org/10.1016/j.ecolecon.2019.106534
  - [Corrected from: Andor, Frondel, & Sommer (2017). The original citation used wrong authors, wrong year, and wrong journal volume. The cited title "Risk Perception of Climate Change: Empirical Evidence for Germany" (Ecological Economics 137) is a different paper entirely.]

* Beatty, T. K. M., Shimshack, J. P., & Volpe, R. J. (2019). Disaster Preparedness and Disaster Response: Evidence from Sales of Emergency Supplies Before and After Hurricanes. *Journal of the Association of Environmental and Resource Economists*, 6(4), 633-668. [NEW — direct behavioral evidence on pre-disaster preparedness]

* Botzen, W. J. W., Kunreuther, H., & Michel-Kerjan, E. (2019). Complementarity and substitutability between flood insurance and self-insurance: Evidence from Hurricane Sandy. *Working paper / Google Scholar*. [NEW — complementarity between structural mitigation and insurance from Sandy homeowners. Verify exact publication venue and year before finalizing.]

* Deryugina, T. (2017). The Fiscal Cost of Hurricanes: Disaster Aid versus Social Insurance. *American Economic Journal: Economic Policy*, 9(3), 168-198. https://doi.org/10.1257/pol.20140296 [NEW — non-disaster transfers dwarf direct disaster aid]

* Gallagher, J. (2014). Learning about an Infrequent Event: Evidence from Flood Insurance Take-Up in the United States. *American Economic Journal: Applied Economics*, 6(3), 206-233. https://doi.org/10.1257/app.6.3.206 [NEW — insurance take-up spikes post-flood, decays over ~9 years]

* Garrett, T. A., & Sobel, R. S. (2003). The Political Economy of FEMA Disaster Payments. *Economic Inquiry*, 41(3), 496-509. https://doi.org/10.1093/ei/cbg023

* Kousky, C., Michel-Kerjan, E. O., & Raschky, P. A. (2018). Does Federal Disaster Assistance Crowd Out Flood Insurance? *Journal of Environmental Economics and Management*, 87, 150-164. https://doi.org/10.1016/j.jeem.2017.05.010

* Peltzman, S. (1975). The Effects of Automobile Safety Regulation. *Journal of Political Economy*, 83(4), 677-725. https://doi.org/10.1086/260352

* Raschky, P. A., & Weck-Hannemann, H. (2007). Charity Hazard — A Real Hazard to Natural Disaster Insurance. *Environmental Hazards*, 7(4), 321-329. https://doi.org/10.1016/j.envhaz.2007.09.002
  - [Note: The 2007 publication is in Environmental Hazards. There is also an Ecological Economics version. Verify which venue the specific claims cited are drawn from.]

* Schneider, S. A., & Kunze, S. (2025). Disastrous Discretion: Political Bias in Relief Allocation Varies Substantially with Disaster Severity. *Review of Economics and Statistics*, 107(5), 1448-1459. https://doi.org/10.1162/rest_a_01319
  - [Corrected from: Gallego et al. (2023). The original citation used entirely wrong authors. Paper circulated as a 2023 working paper; published in REStat 2025.]

* Shabman, L. (2012). Not All Disasters Are Equal: The Reality of FEMA Individual Assistance. *Resources for the Future Discussion Paper*. [Verify exact title and RFF publication number before finalizing.]

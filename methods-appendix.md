# Methods Appendix: Statistical Techniques and Validation Metrics

This appendix provides formal statements, derivations, and implementation details for all econometric methods referenced in the project plan. Techniques are organized by their role in the analysis pipeline.

---

## A. Regression Discontinuity Design

### A.1 Sharp RD Estimand

The sharp RD design identifies a causal effect at a threshold c of a running variable X by exploiting the discontinuity in treatment assignment. Let Y denote the outcome, D the treatment indicator, and X the running variable.

**Setup.** Treatment is assigned deterministically by threshold crossing:

```
D_i = 1[X_i ≥ c]
```

The sharp RD estimand is the difference in conditional expectations at the threshold:

```
τ_SRD = lim_{x↓c} E[Y_i | X_i = x] − lim_{x↑c} E[Y_i | X_i = x]
```

Under the continuity assumption — that the conditional expectation functions E[Y(0)|X=x] and E[Y(1)|X=x] are continuous at x = c — this identifies the average treatment effect at the cutoff:

```
τ_SRD = E[Y_i(1) − Y_i(0) | X_i = c]
```

**Reference:** Hahn, Todd, and van der Klaauw (2001), "Identification and Estimation of Treatment Effects with a Regression-Discontinuity Design," *Econometrica* 69(1), 201-209.

### A.2 Fuzzy RD Estimand

In the fuzzy RD design, treatment assignment is not deterministic at the threshold. Let D_i ∈ {0,1} denote actual treatment receipt (IA declaration status). The probability of treatment jumps discontinuously at the threshold but does not change from 0 to 1:

```
lim_{x↓c} E[D_i | X_i = x] ≠ lim_{x↑c} E[D_i | X_i = x]
```

but neither limit equals 0 or 1.

The fuzzy RD estimand is:

```
τ_FRD = [lim_{x↓c} E[Y_i | X_i = x] − lim_{x↑c} E[Y_i | X_i = x]]
         ÷ [lim_{x↓c} E[D_i | X_i = x] − lim_{x↑c} E[D_i | X_i = x]]
```

This is the ratio of the reduced-form discontinuity in the outcome to the first-stage discontinuity in treatment. Under a local monotonicity assumption (no "defiers" at the threshold), τ_FRD identifies the local average treatment effect (LATE) for compliers at the cutoff — units whose treatment status is changed by crossing the threshold.

**Application in this project:**
- **Mechanism B (contemporaneous RD):** X = TTR-adjusted per-capita damage; c = published FEMA threshold; D = IA declaration status. The first-stage denominator captures the jump in declaration probability at the threshold. The reduced-form numerator captures the jump in post-disaster outcomes.
- **Mechanism A (historical IV):** X = TTR-adjusted per-capita damage for the *prior* event; c = threshold in year t-k; D = prior IA receipt. The second stage uses predicted prior IA receipt as an instrument for current expectations. This is a fuzzy RD used as a first stage for a 2SLS estimator (see Section B).

**Reference:** Imbens and Lemieux (2008), "Regression Discontinuity Designs: A Guide to Practice," *Journal of Econometrics* 142(2), 615-635.

### A.3 Local Linear Regression Estimation

We estimate the RD treatment effect using local linear regression, fitting separate linear functions on each side of the cutoff within a bandwidth h:

```
min_{α_l, β_l} Σ_{i: X_i < c} [Y_i − α_l − β_l(X_i − c)]² · K((X_i − c)/h)

min_{α_r, β_r} Σ_{i: X_i ≥ c} [Y_i − α_r − β_r(X_i − c)]² · K((X_i − c)/h)
```

where K(·) is a kernel function. The treatment effect estimate is:

```
τ̂_SRD = α̂_r − α̂_l
```

The intercept difference captures the discontinuity at x = c. Local linear regression is preferred over local constant (Nadaraya-Watson) because it has superior boundary bias properties — the bias at the boundary point c is of order h² rather than h.

**Kernel choice.** We use the triangular kernel:

```
K(u) = (1 − |u|) · 1[|u| ≤ 1]
```

The triangular kernel is MSE-optimal among non-negative kernels for boundary estimation (Cheng, Fan, and Marron 1997). It gives maximum weight to observations closest to the cutoff and zero weight to observations outside the bandwidth.

**Combining into a single regression.** Equivalently, local linear RD can be estimated via weighted least squares on the pooled sample:

```
Y_i = α + τ · D_i + β₁(X_i − c) + β₂ · D_i · (X_i − c) + ε_i
```

with weights K((X_i − c)/h). The coefficient τ on the treatment dummy is the RD estimate. The interaction D_i · (X_i − c) allows different slopes on each side of the cutoff.

For the fuzzy RD, this becomes a 2SLS regression where D_i is instrumented by the threshold indicator 1[X_i ≥ c]:

**First stage:**
```
D_i = π₀ + π₁ · 1[X_i ≥ c] + π₂(X_i − c) + π₃ · 1[X_i ≥ c] · (X_i − c) + v_i
```

**Second stage:**
```
Y_i = α + τ · D̂_i + β₁(X_i − c) + β₂ · D̂_i · (X_i − c) + ε_i
```

Both stages use kernel weights K((X_i − c)/h). The coefficient τ is the fuzzy RD estimate.

**Reference:** Fan and Gijbels (1996), *Local Polynomial Modelling and Its Applications*, Chapman & Hall.

### A.4 Bandwidth Selection: Imbens-Kalyanaraman (IK) and Calonico-Cattaneo-Titiunik (CCT)

#### A.4.1 IK Optimal Bandwidth

Imbens and Kalyanaraman (2012) derive the MSE-optimal bandwidth for local linear RD estimation. The bandwidth minimizes the asymptotic mean squared error, which trades off squared bias (decreasing in h) against variance (increasing in h).

The IK optimal bandwidth is:

```
h_IK = C_K · [σ²(c) / (n · f_X(c) · (m₊''(c) − m₋''(c))²)]^(1/5)
```

where:
- C_K is a constant depending on the kernel (C_K = 3.4375 for the triangular kernel)
- σ²(c) is the conditional variance of Y at the cutoff (combining left and right limits)
- n is the sample size
- f_X(c) is the density of X at the cutoff
- m₊''(c), m₋''(c) are the second derivatives of the conditional expectation functions from the right and left

In practice, σ²(c), f_X(c), and the second derivatives are estimated using pilot bandwidths and regularization procedures detailed in Imbens and Kalyanaraman (2012).

**Reference:** Imbens and Kalyanaraman (2012), "Optimal Bandwidth Choice for the Regression Discontinuity Estimator," *Review of Economic Studies* 79(3), 933-959.

#### A.4.2 CCT Robust Bandwidth and Inference

Calonico, Cattaneo, and Titiunik (2014) show that conventional RD confidence intervals based on the IK bandwidth undercover because they ignore the bias inherent in local polynomial estimation. They propose a bias-corrected estimator with robust standard errors.

The bias-corrected estimator adjusts the local linear estimate by subtracting an estimate of the leading bias term:

```
τ̂_bc = τ̂_h − h² · B̂
```

where B̂ is estimated from local quadratic regressions using a separate, larger bandwidth b > h. The robust confidence interval accounts for the additional variance introduced by the bias correction.

The CCT procedure simultaneously selects both bandwidths (h for estimation, b for bias correction) and provides confidence intervals with correct coverage.

**Implementation:** The `rdrobust` Python/R package implements both IK and CCT procedures. We report CCT bias-corrected estimates with robust confidence intervals as the primary specification, with conventional estimates at the IK bandwidth as a robustness check.

**Reference:** Calonico, Cattaneo, and Titiunik (2014), "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs," *Econometrica* 82(6), 2295-2326.

#### A.4.3 Bandwidth Robustness

Following standard practice, we report estimates at multiple bandwidths: 0.5h*, 0.75h*, h*, 1.25h*, 1.5h*, and 2h*, where h* is the data-driven optimal bandwidth. If the point estimate is stable across bandwidths, the result is not an artifact of bandwidth choice.

---

## B. Two-Stage Least Squares with RD First Stage (Mechanism A)

Mechanism A uses a fuzzy RD at the *historical* threshold as the first stage for a 2SLS estimator of the effect of prior IA receipt on current mitigation behavior.

### B.1 Formal Setup

Let i index ZIPs and t index the current disaster event. Let t-k denote the most recent prior event in the same county.

**Endogenous variable:** D_{i,t-k} = 1[ZIP i received IA in prior event t-k]

**Instrument:** Z_{i,t-k} = 1[RunningVar_{c,t-k} > 0] (threshold crossing in prior event)

**Running variable control:** f(RunningVar_{c,t-k}) — flexible polynomial in the prior event's TTR-adjusted per-capita damage relative to threshold

**Outcome:** Y_{i,t} — mitigation proxy in the current event (contents damage ratio, damage per flood foot, etc.)

**First stage:**
```
D_{i,t-k} = π₀ + π₁ · Z_{i,t-k} + f(RunningVar_{c,t-k}) + X_{i,t}γ + v_{i,t}
```

**Second stage:**
```
Y_{i,t} = β₀ + β₁ · D̂_{i,t-k} + f(RunningVar_{c,t-k}) + X_{i,t}δ 
          + HazardIntensity_{i,t}λ + WarningLeadTime_{i,t}θ + ε_{i,t}
```

where X_{i,t} includes ZCTA demographics, year FE, state FE, hazard-type FE, and election cycle indicators.

**Interpretation of β₁:** Under the identifying assumptions (Section A.2), β₁ is the LATE of prior IA receipt on current mitigation behavior for compliers — ZIPs whose prior IA receipt was determined by the threshold crossing. This is a two-sided test:
- β₁ < 0: Peltzman/substitution (prior IA reduces current mitigation)
- β₁ > 0: Andor/complementarity (prior IA increases current mitigation)
- β₁ = 0: no effect

### B.2 First-Stage Diagnostics

The first-stage F-statistic tests the null that the instrument Z has no effect on the endogenous variable D:

```
F = (π̂₁ / se(π̂₁))²
```

Under the Staiger and Stock (1997) rule of thumb, F < 10 indicates a weak instrument problem. With a weak instrument, 2SLS estimates are biased toward OLS and confidence intervals have incorrect coverage.

We report the effective F-statistic of Olea and Pflueger (2013), which is robust to heteroskedasticity and clustering:

```
F_eff = (π̂₁' · Var(Z)⁻¹ · π̂₁) / k
```

where k is the number of instruments (k=1 in our case, so F_eff reduces to the heteroskedasticity-robust t-statistic squared).

**Diagnostic plan:**
1. Report F_eff for the full Mechanism A sample
2. Report F_eff stratified by years-since-prior bin (0-2, 3-5, 6-9, 10+) — instrument power should decay with temporal distance from the prior event (Gallagher 2014 benchmark: ~9-year half-life)
3. Report F_eff separately for flood-only and all-hazard samples
4. Report F_eff separately for election-year vs. non-election-year prior events
5. If F_eff < 10 in the primary sample, report Anderson-Rubin confidence intervals (valid under weak instruments)

**References:**
- Staiger and Stock (1997), "Instrumental Variables Regression with Weak Instruments," *Econometrica* 65(3), 557-586.
- Olea and Pflueger (2013), "A Robust Test for Weak Instruments," *Journal of Business & Economic Statistics* 31(3), 358-369.

### B.3 Anderson-Rubin Weak-Instrument-Robust Inference

If the first-stage F-statistic is below 10, standard 2SLS inference is unreliable. The Anderson-Rubin (AR) test provides confidence intervals that are valid regardless of instrument strength.

The AR test statistic for H₀: β₁ = β₁⁰ is constructed by substituting the null value into the structural equation and testing whether the resulting residual is uncorrelated with the instrument:

```
Y_{i,t} − β₁⁰ · D_{i,t-k} = α + f(RunningVar) + X·δ + η_i
```

Under H₀, the coefficient on Z_{i,t-k} in a regression of η̂ on Z (controlling for f(RunningVar) and X) should be zero. The AR confidence set is the collection of β₁⁰ values for which this F-test fails to reject at the chosen significance level.

The AR confidence set may be empty (no β₁ consistent with the data), unbounded (instrument too weak to identify the parameter), or a proper interval.

**Reference:** Anderson and Rubin (1949), "Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations," *Annals of Mathematical Statistics* 20(1), 46-63.

---

## C. Validation and Diagnostic Tests

### C.1 McCrary Density Test

The McCrary test detects manipulation of the running variable at the threshold. If agents can precisely sort above or below the cutoff (e.g., FEMA or states gaming damage estimates), the density of X will be discontinuous at c. Under no manipulation, the density f_X(x) is continuous at x = c.

**Null hypothesis:** f_X(x) is continuous at x = c.

**Test procedure (McCrary 2008):**

1. Construct a histogram of the running variable using bins of width b on each side of the cutoff, with a bin boundary exactly at c.

2. Let f̂_j denote the normalized frequency in bin j. Estimate local linear regressions of f̂_j on the bin midpoints separately on each side of c:

```
f̂_j = α_l + β_l · m_j + u_j     for m_j < c
f̂_j = α_r + β_r · m_j + u_j     for m_j ≥ c
```

using a triangular kernel with bandwidth h.

3. The test statistic is:

```
θ̂ = ln(α̂_r) − ln(α̂_l)
```

The log-difference of the estimated densities at the cutoff. Under H₀, θ̂ is asymptotically normal with variance estimated from the local linear fits:

```
T = θ̂ / se(θ̂) → N(0,1)
```

A significantly positive θ̂ indicates excess mass just above the cutoff (manipulation upward); significantly negative indicates excess mass just below.

**Implementation:** Use the `rddensity` package (Cattaneo, Jansson, and Ma 2020), which improves on the original McCrary procedure with robust bias-corrected inference and data-driven bandwidth selection.

**Application:** Run on both the historical running variable (Mechanism A first stage) and the contemporaneous running variable (Mechanism B). If the McCrary test rejects at the 5% level, the threshold is likely subject to manipulation and the RD design is compromised.

**References:**
- McCrary (2008), "Manipulation of the Running Variable in the Regression Discontinuity Design: A Density Test," *Journal of Econometrics* 142(2), 698-714.
- Cattaneo, Jansson, and Ma (2020), "Simple Local Polynomial Density Estimators," *Journal of the American Statistical Association* 115(531), 1449-1455.

### C.2 Placebo RD Tests on Predetermined Covariates

If the threshold is quasi-random, then predetermined covariates — characteristics determined before the running variable is realized — should not show discontinuities at the cutoff. Finding a significant jump in a predetermined covariate suggests that the threshold is correlated with observable (and potentially unobservable) characteristics, violating the continuity assumption.

**Procedure:** For each predetermined covariate W (e.g., median household income, presidential vote margin, pre-event NFIP take-up rate, housing age distribution), estimate the standard RD specification with W as the dependent variable:

```
W_i = α + τ_W · 1[X_i ≥ c] + β₁(X_i − c) + β₂ · 1[X_i ≥ c] · (X_i − c) + ε_i
```

with kernel weights and the same bandwidth used in the main analysis. Under the null of valid RD, τ_W = 0 for all predetermined covariates.

**Multiple testing correction:** With k covariates tested, the probability of at least one false rejection at the 5% level is 1 − (0.95)^k. We apply a Bonferroni correction, testing each covariate at the α/k level. With k = 5 covariates, the adjusted significance level is 1%.

Additionally, we report a joint F-test of the null that all covariate discontinuities are simultaneously zero, estimated by stacking all covariates and testing the joint significance of the treatment indicators.

**Motivation from Schneider and Kunze (2025):** Political bias in disaster declarations concentrates in medium-intensity events. If the presidential vote margin shows a discontinuity at the damage threshold, political sorting is contaminating the design. This is a stronger test than McCrary because it detects sorting on specific political variables even when the overall density is smooth.

### C.3 Placebo Threshold Tests

If the treatment effect is truly driven by the threshold, then estimating the RD at placebo thresholds (values of c where no policy discontinuity exists) should yield null results.

**Procedure:** For a set of placebo thresholds c₁, c₂, ..., c_m located at quantiles of the running variable away from the true threshold, estimate the standard RD at each c_j and report the distribution of placebo estimates. The true-threshold estimate should be larger (in absolute value) than the placebo estimates.

**Implementation:** Choose placebo thresholds at the 10th, 25th, 75th, and 90th percentiles of the running variable distribution. Estimate the RD at each and report the implied permutation p-value: the fraction of placebo estimates that exceed the true-threshold estimate.

### C.4 Donut RD

If there are concerns about precise manipulation very close to the threshold (e.g., damage estimates nudged just above the cutoff), the donut RD excludes observations within a small window δ around the cutoff:

```
Estimation sample: {i : |X_i − c| > δ}
```

This removes potentially manipulated observations while still exploiting the discontinuity from observations further from the cutoff.

**Procedure:** Estimate the standard local linear RD on the donut sample for δ ∈ {0.05, 0.10, 0.15, 0.20} (in per-capita damage units). If estimates are stable across donut widths, manipulation near the boundary is unlikely to explain the results.

**Reference:** Barreca, Lindo, and Waddell (2016), "Heaping-Induced Bias in Regression-Discontinuity Designs," *Economic Inquiry* 54(1), 268-293.

---

## D. Synthetic Difference-in-Differences (SDID)

### D.1 Setup and Estimand

Consider a panel of N units (ZIPs) observed over T periods (disaster events and inter-event periods). Let Y_{it} denote the outcome for unit i in period t. The last T_post periods are the post-treatment period. Units i = 1, ..., N_tr are treated; units i = N_tr+1, ..., N are controls.

The potential outcomes model is:

```
Y_{it} = Y_{it}(0) + D_{it} · τ_{it}
```

where D_{it} = 1 for treated units in post-treatment periods and Y_{it}(0) is the untreated potential outcome. The target estimand is:

```
τ = (1/N_tr · T_post) Σ_{i≤N_tr} Σ_{t>T-T_post} τ_{it}
```

the average treatment effect on the treated (ATT) over the post-treatment period.

### D.2 SDID Estimator

The SDID estimator (Arkhangelsky, Athey, Hirshberg, Imbens, and Peel 2021) constructs both unit weights ω̂ and time weights λ̂ to create a doubly-reweighted DiD estimator:

```
(τ̂_sdid, μ̂, α̂, β̂) = argmin_{τ,μ,α,β} Σ_i Σ_t (Y_{it} − μ − α_i − β_t − D_{it}τ)² · ω̂_i · λ̂_t
```

where:
- α_i are unit fixed effects
- β_t are time fixed effects
- ω̂_i are unit weights (on control units only; treated units receive weight 1/N_tr)
- λ̂_t are time weights (on pre-treatment periods only; post-treatment periods receive weight 1/T_post)

#### D.2.1 Unit Weights

The unit weights ω̂ are chosen to make the weighted average of pre-treatment outcomes for control units match the pre-treatment average for treated units, analogous to synthetic control weights. Formally:

```
ω̂ = argmin_{ω≥0, Σω=1} Σ_{t≤T-T_post} (Σ_{i>N_tr} ω_i Y_{it} − (1/N_tr)Σ_{i≤N_tr} Y_{it})² + ζ² · ‖ω‖²
```

where ζ > 0 is a regularization parameter that prevents exact fitting. The regularization ensures ω̂ has full support when possible, improving variance properties relative to classical SCM.

The regularization parameter is set to:

```
ζ² = (N_tr · T_post) · σ̂²_ε
```

where σ̂²_ε is a variance estimate from the control group.

#### D.2.2 Time Weights

The time weights λ̂ are chosen to make the weighted average of pre-treatment outcomes match the post-treatment average, using only control units:

```
λ̂ = argmin_{λ≥0, Σλ=1} Σ_{i>N_tr} (Σ_{t≤T-T_post} λ_t Y_{it} − (1/T_post)Σ_{t>T-T_post} Y_{it})²
```

This selects pre-treatment periods that are most predictive of post-treatment outcomes, improving efficiency.

### D.3 Inference

Standard errors for τ̂_sdid are computed using a placebo variance estimator. The idea is to estimate placebo treatment effects by iteratively removing one treated unit and applying the estimator, then using the variance of these placebo estimates.

For each treated unit j = 1, ..., N_tr, compute the leave-one-out placebo estimate τ̂^(-j) and estimate:

```
V̂_sdid = (1/(N_tr-1)) Σ_j (τ̂^(-j) − τ̄)²
```

where τ̄ is the average of the placebo estimates. Alternatively, for settings with few treated units, Arkhangelsky et al. (2021) propose a conformal inference approach that provides finite-sample valid p-values without distributional assumptions.

### D.4 Relationship to DiD and SCM

SDID nests both standard DiD and SCM as special cases:
- **DiD:** Set ω̂_i = 1/N_co (equal weights on controls) and λ̂_t = 1/T_pre (equal weights on pre-treatment periods). No reweighting.
- **SCM:** Set λ̂_t = 1/T_pre (equal time weights) and solve for ω̂ without regularization. No intercept shift.

SDID improves on both by allowing the estimator to adapt along both dimensions:
- Relative to DiD: ω̂ adjusts for pre-treatment imbalance when parallel trends may not hold exactly
- Relative to SCM: λ̂ adjusts for time periods that are not equally informative about the treatment effect; the regularized weights avoid overfitting

### D.5 Application in This Project

**Mechanism A:** Treatment = prior IA receipt at the ZIP level. Pre-treatment periods = pre-prior-disaster outcomes. Post-treatment period = current disaster event outcomes. Unit weights match control ZIPs to treated ZIPs on pre-treatment mitigation patterns.

**Mechanism B:** Treatment = IA declaration at the county level, applied to all ZIPs in declared counties. Pre-treatment = periods before the current declaration. Post-treatment = post-declaration period. Unit weights match non-declared-county ZIPs to declared-county ZIPs.

**Software:** `synthdid` R package via `rpy2`. Callaway and Sant'Anna (2021) `did` package as a robustness alternative for the staggered adoption structure.

**Reference:** Arkhangelsky, Athey, Hirshberg, Imbens, and Peel (2021), "Synthetic Difference-in-Differences," *American Economic Review* 111(12), 4088-4118.

---

## E. TTR-Adjusted Running Variable

### E.1 Motivation

FEMA's Preliminary Damage Assessment process considers state fiscal capacity when recommending IA declarations. A state with high Total Taxable Resources (TTR) is expected to absorb more damage before federal intervention. Failing to adjust the running variable for TTR introduces selection bias: high-TTR states appear farther below the threshold than they effectively are, and low-TTR states appear closer.

### E.2 Three Specifications

Let PCD_{c,t} denote per-capita damage in county c during event t, Threshold_t the published FEMA threshold for year t, and TTR_{s,t} the state-level TTR for state s in year t (as a percentage of the national average).

**Specification 1 — Linear TTR adjustment:**

```
RunningVar_{c,t}^(1) = (PCD_{c,t} / (TTR_{s,t} / 100)) − Threshold_t
```

Interpretation: $1 of per-capita damage in a state at 80% of national-average TTR is equivalent to $1.25 in a state at 100%.

**Specification 2 — Log TTR adjustment:**

```
RunningVar_{c,t}^(2) = PCD_{c,t} · ln(100 / TTR_{s,t}) − Threshold_t
```

Allows diminishing marginal effect of fiscal capacity. States near the national average see little adjustment; states far from the average see larger adjustments.

**Specification 3 — Rank TTR adjustment:**

```
RunningVar_{c,t}^(3) = PCD_{c,t} · [1 + (1 − R_{s,t})] − Threshold_t
```

where R_{s,t} is the percentile rank of state s's TTR in year t (R = 1 for the highest-TTR state, R = 0 for the lowest). Bottom-quartile states receive approximately 1.75× weight; top-quartile states receive approximately 1.25× weight.

### E.3 Specification Selection

We select the preferred specification based on first-stage strength: the TTR adjustment that produces the highest effective F-statistic for the threshold → declaration first stage is selected as the primary running variable. The other two are reported as robustness. If all three produce similar first-stage F-statistics, the linear specification is preferred for interpretability.

---

## F. Election Cycle Diagnostics

### F.1 Motivation

Schneider and Kunze (2025) show that political bias in FEMA disaster declarations is hump-shaped in disaster severity: concentrated in medium-intensity events where the per-capita threshold operates. Declarations are unbiased for very strong or very weak events, but areas governed by the president's co-partisans receive up to twice as many declarations for medium-intensity hurricanes.

### F.2 Diagnostic Test

**First-stage interaction model:**

```
D_{c,t} = π₀ + π₁ · 1[X_{c,t} ≥ c] + π₂ · ElectionYear_t 
          + π₃ · 1[X_{c,t} ≥ c] × ElectionYear_t
          + f(X_{c,t}) + State_s + ε_{c,t}
```

If π₃ is large and significant, the threshold has more predictive power for declarations in election years, suggesting political sorting contaminates the first stage.

**Decision rule:** If the first-stage F-statistic in election years exceeds 2× the F-statistic in non-election years, flag political contamination. Report non-election-year subsample as robustness. If the full-sample results are qualitatively different from the non-election-year results, the main estimates may reflect political dynamics rather than the damage threshold.

A parallel test replaces ElectionYear with GovernorPresidentAlignment (indicator for same party). A parallel test uses the flood-only subsample, where Schneider and Kunze's hurricane-specific political bias is not operative.

---

## G. Temporal Decay Analysis

### G.1 Motivation

Gallagher (2014) found that NFIP flood insurance take-up spikes the year after a flood and decays back to baseline over approximately 9 years, consistent with a Bayesian learning model with forgetting. If the Mechanism A effect operates through an expectations channel, it should decay at a comparable rate. If it does not decay, infrastructure or institutional channels (rebuilt to code, changed drainage) may dominate.

### G.2 Specification

**Binned temporal decay:**

```
Y_{i,t} = β₀ + Σ_k β_k · D̂_{i,t-k} · 1[k ∈ Bin_k] + f(RunningVar) + X·δ + ε
```

where Bin_k ∈ {[0,2], [3,5], [6,9], [10,15]} indexes years since the prior IA event, and D̂_{i,t-k} is the instrumented prior IA indicator.

The sequence {β₁, β₂, β₃, β₄} traces the decay curve. Under the expectations hypothesis, we expect |β₁| > |β₂| > |β₃| > |β₄| ≈ 0.

**First-stage power by bin:** Report the effective F-statistic separately for each temporal bin. If F < 10 in the 10+ year bin, the instrument has lost relevance and that bin should be excluded from the primary sample.

**Comparison to Gallagher benchmark:** Compute the implied half-life of the Mechanism A effect (the years-since-prior value at which the effect reaches half its 0-2 year magnitude) and compare to Gallagher's ~9-year estimate. If the decay rates are similar, the expectations channel is the likely mechanism. If the Mechanism A effect decays much faster (half-life < 3 years), availability bias rather than Bayesian updating may drive the result. If it decays much slower or not at all, non-expectations channels dominate.

---

## H. Tenure Heterogeneity Interaction

### H.1 Motivation

Andor, Osberghaus, and Simora (2020) found that physical mitigation and expected aid are complements in German households. Botzen, Kunreuther, and Michel-Kerjan (2019) found complementarity for structural precautionary measures but substitution for emergency measures among Sandy homeowners. Both studies suggest the moral hazard response varies by the type of protective action and the economic incentives of the actor.

In the FEMA context, homeowners and renters face different incentive structures:
- **Owners** are eligible for full IHP housing assistance (repair/replacement) and have incentives to mitigate structural damage (they bear the residual loss beyond IA limits)
- **Renters** are only eligible for Other Needs Assistance (ONA) and have no incentive to mitigate structural damage but strong incentive to protect contents (personal property)

### H.2 Specification

**Interaction model:**

```
Y_{i,t} = β₀ + β₁ · D̂_{i,t-k} + β₂ · HomeownerRate_i 
          + β₃ · D̂_{i,t-k} × HomeownerRate_i + Controls + ε_{i,t}
```

where HomeownerRate_i is the ZCTA-level homeownership rate from ACS table B25003 (owner-occupied / total occupied housing units).

**Interpretation of β₃:**
- For the contents damage ratio outcome: β₃ > 0 means that in high-homeownership ZIPs, prior IA receipt is associated with relatively higher contents damage (less contents mitigation by owners, who focus on structural hardening). β₃ < 0 means owners with IA history do more contents mitigation than renters with IA history.
- The signs of β₁ and β₃ together reveal whether the Andor/Botzen mitigation-type heterogeneity replicates in the U.S. federal aid context.

### H.3 Additional Decomposition

Where data allow (HousingAssistanceOwners and HousingAssistanceRenters are separate datasets), compute outcomes separately for owner-occupied and renter-occupied properties within each ZIP and estimate the effect on each subgroup directly, rather than relying on the ZCTA-level interaction as a proxy.

---

## I. Permutation Inference for SDID

### I.1 Motivation

With a moderate number of treated units and potential violations of distributional assumptions, asymptotic inference may be unreliable. Permutation-based inference provides finite-sample valid p-values.

### I.2 Procedure

1. For the observed treatment assignment T = {treated counties}, estimate τ̂_sdid.
2. For each of K random reassignments T_k (drawing N_tr counties from the control pool without replacement), estimate the placebo τ̂_k.
3. The permutation p-value for a two-sided test is:

```
p = (1/K) Σ_k 1[|τ̂_k| ≥ |τ̂_sdid|]
```

4. Set K = 1000 for computational feasibility.

**Refinement:** Fisher's exact p-value uses all (N choose N_tr) possible reassignments, which is infeasible for large N. The random permutation approximation introduces simulation error bounded by 1/√K.

**Reference:** Fisher (1935), *The Design of Experiments*, Oliver & Boyd.

---

## J. Multiple Outcome Adjustment

### J.1 Motivation

The project tests multiple outcome variables for each mechanism (4 for Mechanism A, 7 for Mechanism B). Without correction, the probability of at least one false positive at α = 0.05 reaches 1 − (0.95)^k, which is 0.30 for k = 7.

### J.2 Corrections

**Bonferroni correction (conservative):** Test each outcome at α/k. With k = 4 (Mechanism A), the adjusted significance level is 0.0125. With k = 7 (Mechanism B), it is 0.0071.

**Holm-Bonferroni (step-down, less conservative):** Order the k p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(k). Reject H_(j) if p_(j) ≤ α/(k - j + 1) for all j' ≤ j.

**Benjamini-Hochberg (FDR control):** Order the k p-values. Reject H_(j) if p_(j) ≤ (j/k) · α. This controls the false discovery rate at α rather than the family-wise error rate, and is less conservative than Bonferroni.

**Reporting strategy:** Report unadjusted p-values for each outcome, plus Holm-Bonferroni adjusted p-values for the family-wise error rate and Benjamini-Hochberg adjusted p-values for the FDR. Clearly label which corrections are applied.

**References:**
- Bonferroni (1936), "Teoria statistica delle classi e calcolo delle probabilità," *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*.
- Benjamini and Hochberg (1995), "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing," *Journal of the Royal Statistical Society, Series B* 57(1), 289-300.

---

## K. Power Analysis

### K.1 Motivation

The effective sample for Mechanism A — ZIPs with prior near-threshold events within 10 years, current events with ≥24h warning, and positive NFIP/IA claims — may be small. Power analysis before committing to specifications determines whether the design can detect economically meaningful effects.

### K.2 Minimum Detectable Effect (MDE)

For a two-sided test at significance level α with power 1 − κ, the minimum detectable effect in a fuzzy RD with first-stage coefficient π₁ is approximately:

```
MDE = (z_{α/2} + z_κ) · σ_Y / (π₁ · √(n_eff))
```

where:
- z_{α/2} and z_κ are standard normal quantiles (z_{0.025} = 1.96, z_{0.20} = 0.84 for 80% power)
- σ_Y is the standard deviation of the outcome variable
- π₁ is the first-stage coefficient (jump in treatment probability at threshold)
- n_eff is the effective sample size within the bandwidth

The effective sample size for local linear RD with a triangular kernel is approximately:

```
n_eff ≈ n · h · f_X(c) · (4/3)
```

where the (4/3) factor reflects the variance-inflation from kernel weighting.

### K.3 Implementation

After data construction (Step 15), compute n_eff, σ_Y (for each outcome), and π̂₁ (from the first-stage diagnostic) to calculate the MDE. Compare the MDE to economically meaningful benchmarks:

- For the contents damage ratio: is the MDE smaller than 0.05 (a 5 percentage point change in the contents share)?
- For damage per flood foot: is the MDE smaller than $5,000/foot (a ~10% change at the mean)?
- For IA application rate: is the MDE smaller than 0.02 (a 2 percentage point change)?

If the MDE exceeds economically meaningful thresholds, the design is underpowered and the bandwidth or sample restrictions should be reconsidered.

**Reference:** Cattaneo, Titiunik, and Vazquez-Bare (2019), "Power Calculations for Regression-Discontinuity Designs," *Stata Journal* 19(1), 210-245.

---

## References

Anderson, T. W., and Rubin, H. (1949). "Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations." *Annals of Mathematical Statistics* 20(1), 46-63.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., and Peel, S. (2021). "Synthetic Difference-in-Differences." *American Economic Review* 111(12), 4088-4118.

Barreca, A. I., Lindo, J. M., and Waddell, G. R. (2016). "Heaping-Induced Bias in Regression-Discontinuity Designs." *Economic Inquiry* 54(1), 268-293.

Benjamini, Y., and Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society, Series B* 57(1), 289-300.

Calonico, S., Cattaneo, M. D., and Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." *Econometrica* 82(6), 2295-2326.

Callaway, B., and Sant'Anna, P. H. C. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225(2), 200-230.

Cattaneo, M. D., Jansson, M., and Ma, X. (2020). "Simple Local Polynomial Density Estimators." *Journal of the American Statistical Association* 115(531), 1449-1455.

Cattaneo, M. D., Titiunik, R., and Vazquez-Bare, G. (2019). "Power Calculations for Regression-Discontinuity Designs." *Stata Journal* 19(1), 210-245.

Cheng, M. Y., Fan, J., and Marron, J. S. (1997). "On Automatic Boundary Corrections." *Annals of Statistics* 25(4), 1691-1708.

Fan, J., and Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.

Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.

Gallagher, J. (2014). "Learning about an Infrequent Event: Evidence from Flood Insurance Take-Up in the United States." *American Economic Journal: Applied Economics* 6(3), 206-233.

Hahn, J., Todd, P., and van der Klaauw, W. (2001). "Identification and Estimation of Treatment Effects with a Regression-Discontinuity Design." *Econometrica* 69(1), 201-209.

Imbens, G. W., and Kalyanaraman, K. (2012). "Optimal Bandwidth Choice for the Regression Discontinuity Estimator." *Review of Economic Studies* 79(3), 933-959.

Imbens, G. W., and Lemieux, T. (2008). "Regression Discontinuity Designs: A Guide to Practice." *Journal of Econometrics* 142(2), 615-635.

McCrary, J. (2008). "Manipulation of the Running Variable in the Regression Discontinuity Design: A Density Test." *Journal of Econometrics* 142(2), 698-714.

Olea, J. L. M., and Pflueger, C. (2013). "A Robust Test for Weak Instruments." *Journal of Business & Economic Statistics* 31(3), 358-369.

Schneider, S. A., and Kunze, S. (2025). "Disastrous Discretion: Political Bias in Relief Allocation Varies Substantially with Disaster Severity." *Review of Economics and Statistics* 107(5), 1448-1459.

Staiger, D., and Stock, J. H. (1997). "Instrumental Variables Regression with Weak Instruments." *Econometrica* 65(3), 557-586.

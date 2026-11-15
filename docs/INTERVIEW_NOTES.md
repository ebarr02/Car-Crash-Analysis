# Explaining this project in an actuarial interview

## A 60-second explanation

“I built a Python and SQL motor insurance expected-loss study using 678,013 policies. I reconciled reported claim counts with the positive claim-amount records, modeled annual claim frequency with exposure weights, and modeled positive severity separately. I compared 12 frequency–severity combinations using shared policy-level cross-validation and selected Poisson boosting with constant severity. It improved internal-holdout Tweedie deviance by 6.8% versus a flat loss-rate baseline. I also checked calibration, bootstrapped uncertainty, and simulated aggregate losses under explicit assumptions. The most important finding was that better standalone severity fit did not automatically improve combined pricing, and large claims made uncertainty material.”

Use the numbers in the executed report. They describe this configuration and dataset, not guaranteed future performance. Be ready to explain the formulas and code before using the project on a résumé.

## Suggested résumé entry

**Motor Insurance Expected-Loss Modeling | Python, SQL, scikit-learn**

- Built an exposure-aware frequency–severity analysis of 678K policies, comparing 12 model combinations through policy-level cross-validation; improved internal-holdout Tweedie deviance by 6.8% versus a flat loss-rate baseline.
- Reconciled claim records, audited 195 unmatched records, and assessed portfolio uncertainty using 20,000-draw aggregate-loss simulations, stress scenarios and paired-policy bootstrap intervals.

Optional replacement bullet if the role emphasizes automation: “Implemented restartable analysis with saved policy splits, model artifacts, run manifests and automated checks for exposure calculations, joins and simulation moments.”

These describe implemented project work. Do not claim commercial deployment, premium savings, production pricing approval, regulatory capital calculation or independent external validation.

## Questions you should be able to answer

**Why include exposure?** A policy observed for six months offers less time to generate claims than a comparable full-year policy. Frequency is an annual rate; expected coverage-period count equals that rate times exposure. The model fits counts/exposure using exposure weights. In an unpenalized Poisson GLM this matches a log-exposure offset, which the tests verify.

**Why model frequency and severity separately?** They explain different parts of expected loss and permit different predictors/effects. The product estimates annual expected loss. A direct Tweedie model is a reasonable future comparator; it is not currently fitted here.

**Why use positive-record counts rather than ClaimNb?** The reported count and severity records do not always reconcile. Positive-record frequency times severity of those same records gives a coherent expected-cost target. This changes the estimand; it does not prove the discrepancy is fixed or explain whether unmatched/absent records represent unpaid, missing or differently defined claims.

**What did SQL contribute?** Raw table storage, unique policy validation, claims-to-policy reconciliation, aggregate loss construction and a queryable results database. A many-to-many join would inflate losses, so policy uniqueness is checked before joining.

**Why a Poisson model?** It has a positive log-linked mean suited to nonnegative counts and an interpretable exposure treatment. Extra count variance would affect distributional uncertainty; Poisson loss can still be a useful way to estimate a conditional mean. This project does not demonstrate that Poisson counts fully describe actual variance.

**Why a Gamma model?** Severity is positive, right-skewed, and its mean is modeled with a log link. A Gamma loss/model is a candidate, not proof that the extreme tail is Gamma distributed. The project fits arithmetic claim amounts directly.

**Why did constant severity win the combined model?** The validation objective is policy-level expected cost. Standalone Gamma deviance evaluates individual positive-claim predictions. Noise, large losses and the interaction with frequency mean their rankings can differ. Simpler severity was selected from out-of-fold evidence rather than assumed inferior to a more flexible model.

**How did you prevent leakage?** No ID, target, actual loss or observed claim count enters the predictors. Frequency and severity share policy-level folds. Preprocessing and models fit inside each training fold. Selection occurs using out-of-fold training predictions, followed by refitting on training data. The reused portfolio still makes the final evaluation an internal holdout rather than a wholly new external test.

**Why Tweedie deviance?** It accommodates zeros and positive continuous costs and provides a common expected-loss comparison. Power 1.5 is fixed in advance here. A different power could change model ranking and would need training-only sensitivity assessment.

**Is an actual/expected ratio near one enough?** No. It only checks the total. The second risk decile has actual/expected around 2.05 while the highest has about 1.23 in this run. Large claims and subgroup uncertainty matter; good aggregate calibration can conceal weak segment calibration.

**What does the bootstrap establish?** Conditional on the fitted models, resampling whole holdout policies gives an approximate interval for paired model differences and aggregate calibration. The deviance difference interval is approximately -10.41 to -2.68, favoring the selected model. The actual/expected interval is much wider, approximately 0.83 to 1.19. Neither interval includes refitting, model selection, systematic data issues or future drift.

**Why do large claims matter?** The top 1% of training claims account for about 38.6% of claim cost. Removing them would change the target and make prediction appear easier. The project retains them, reports their influence and shows hypothetical cap sensitivity separately.

**What is the simulation doing?** It draws a Poisson total number of claims and independently samples that many training claim amounts. Its analytical mean is lambda times mean severity and its variance is lambda times the second severity moment. Simulated moments are compared with these values. The empirical-severity simulation is a constant-severity benchmark, not a conditional Gamma simulation.

**What do the simulated percentiles mean?** They describe outcomes under the specified fitted model and assumptions. They exclude dependence, unseen tail events, count overdispersion and estimation uncertainty, so they are not regulatory capital estimates.

**Why 20,000 draws?** It gives roughly 200 draws above the empirical 99th percentile, compared with around 20 under the earlier 2,000-draw version. It reduces Monte Carlo noise, but collecting better data and improving assumptions can matter much more than adding draws.

**What would you do next?** Obtain an independent or time-based dataset, verify quote-time predictor availability and coverage definitions, investigate count discrepancies and count dispersion, add a direct Tweedie comparator, and study conditional/extreme severity and parameter uncertainty. Add expenses and profitability only once the relevant data and business assumptions exist.

## Skills this demonstrates

Actuarial: exposure, frequency/severity decomposition, pure premium, GLMs, calibration, large-loss sensitivity, simulation and judgment about uncertainty.

Analytical: SQL joins/reconciliation, Python pipelines, out-of-fold validation, reproducibility, model comparison, feature diagnostics and communicating limitations.

Engineering: a command-line workflow, saved progress, artifact hashing, reusable scoring, independent financial-invariant tests and offline reporting.

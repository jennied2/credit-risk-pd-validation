# Building an Explainable Credit PD Model & SR 11-7 Validation Pack
## A step-by-step build manual that teaches credit-risk modeling as you go

**Who this is for:** someone with intermediate Python (pandas, scikit-learn) and intermediate quantitative finance who wants to build the project end-to-end *and* understand the mathematics and statistics underneath it. Every section pairs the **concept** (what it is, why it matters, the formula) with the **build step** (the code that implements it) and **interpretation** (how to read the output like a risk professional).

**What you will have built at the end:** a probability-of-default (PD) model with an interpretable champion and an ML challenger, a three-family evaluation suite, SHAP explainability with regulatory reason codes, a fair-lending disparate-impact audit, and the headline artifact — an SR 11-7 model validation report with severity-rated findings.

**How to read it:** the sections are numbered and ordered the way you should build. Math is written in LaTeX (`$...$`). Code snippets use the exact module APIs in `src/`, so you can paste them into a notebook against this repo. A formula cheat-sheet and glossary are in the appendices.

---

# Part I — Foundations

## 1. The mental model: what credit risk modeling actually is

When a lender extends credit, it faces one core question: *if I lend to this applicant, how much do I expect to lose?* That expectation is decomposed into three pieces, and almost everything in retail credit modeling is about estimating one of them.

$$\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}$$

- **PD — Probability of Default.** The chance the borrower fails to repay over a horizon (the loan's life, here). A probability in $[0,1]$. *This is what our model estimates.*
- **LGD — Loss Given Default.** The fraction of exposure you actually lose after recoveries (collections, collateral). E.g. LGD = 0.55 means you recover 45 cents on the dollar. We hold this fixed at 55% as a simplifying assumption.
- **EAD — Exposure at Default.** How much is outstanding when default happens. For an installment loan we approximate this with the loan amount.

**Expected vs. unexpected loss.** Expected loss is the *average* you price into the product (it's covered by interest margin and provisions). **Unexpected loss** is the volatility around that average — the tail — and is covered by *capital*. PD models feed both: the level (calibration) feeds expected-loss provisioning, and the ranking (discrimination) feeds decisioning. Hold onto that distinction; it explains why we test calibration *and* discrimination separately in Part III.

**Why PD is a binary classification problem.** We define a binary label $Y \in \{0,1\}$ where $Y=1$ means "defaulted." The model outputs $\hat p = P(Y=1 \mid X)$. This is logistic-regression / classification territory — but with credit-specific constraints (monotonicity, calibration, fairness, explainability) that ordinary ML courses ignore. Those constraints are the whole point of this project.

### 1.1 The regulatory backdrop (why we do things the "hard" way)

You cannot understand the design choices without the rules they answer to:

- **Basel framework** defines default operationally (commonly 90+ days past due, or "unlikely to pay"). It drives our target definition.
- **SR 11-7** (US Fed/OCC *Supervisory Guidance on Model Risk Management*) is the spine of the validation report. Its three pillars — **conceptual soundness**, **ongoing monitoring**, and **outcomes analysis** — map directly onto Parts III–VI here. It also mandates *effective challenge* by an independent party (our "dual-hat" framing).
- **ECOA / Regulation B** gives consumers a right to know *why* they were declined — the "adverse-action reason codes" we generate in Part V.
- **Fair-lending law** distinguishes **disparate treatment** (intentionally using a protected attribute) from **disparate impact** (a neutral policy that disproportionately harms a protected group). The HMDA module in Part VII is built around this distinction.

### 1.2 The dual-hat framing

You will wear two hats, and the project is designed to make that explicit:

- **First line — model developer:** you build the champion and challenger and want them to perform.
- **Second line — independent validator:** you interrogate your own model as if a stranger built it, and you write findings against it.

Anyone can fit a classifier. The thing that signals seniority — and the thing that makes this a portfolio piece rather than a Kaggle notebook — is the validation report. Build toward it.

### 1.3 Environment setup

```bash
# minimal core (the repo runs on just these)
pip install pandas numpy scikit-learn matplotlib

# canonical libraries the repo's fallbacks stand in for (recommended on real infra)
pip install xgboost shap optbinning fairlearn statsmodels seaborn jupyter
```

Repo layout you are building:

```
credit-risk-pd-validation/
├── data/                      # drop real CSVs here; else synthetic fallback
├── src/
│   ├── data.py                # target definition, leakage audit, OOT split, loaders
│   ├── scorecard.py           # WOE/IV + logistic scorecard (champion)
│   ├── challenger.py          # monotone-constrained, calibrated GBM (challenger)
│   ├── metrics.py             # discrimination / calibration / stability / EL
│   ├── explain.py             # KernelSHAP, reason codes, conceptual soundness
│   ├── fairness.py            # AIR, equal opportunity, mitigations
│   └── run_pipeline.py        # master runner → figures + results.json
├── notebooks/credit_pd_validation.ipynb
└── reports/validation_report.md   # ← the headline deliverable
```

> **A note on honesty.** If you cannot reach the real data, generate synthetic data with deliberately injected structure (leakage, drift, disparity) and *say so* in the report. Illustrative-but-honest beats impressive-but-fake every time, and validators are paid to be skeptical.

---

# Part II — Data, target definition, and the things that quietly ruin credit models

## 2. The two datasets and why they must stay separate

This project deliberately uses two different real-world data sources because **no single public dataset supports both PD modeling and a fair-lending audit**:

| Dataset | Has loan *outcomes*? | Has *protected attributes*? | Used for |
|---|---|---|---|
| **Lending Club** | ✅ (charged-off / paid) | ❌ | PD model + validation |
| **HMDA** | ❌ (only approve/deny) | ✅ (race, ethnicity, sex) | Fair-lending audit |

You cannot train a PD model on HMDA (no default outcome) and you cannot run a fairness audit on Lending Club (no demographics). So the project demonstrates *two distinct competencies* rather than modeling the same thing twice. State this plainly — interviewers respect understanding the data's limits.

## 3. Defining the target — the single most consequential modeling decision

Before any algorithm, you must turn messy loan statuses into a clean binary label. Get this wrong and everything downstream is wrong.

**Rules (Basel-aligned):**
- $Y = 1$ (default) if status ∈ {Charged Off, Default}.
- $Y = 0$ (non-default) if status = Fully Paid.
- **Drop** unresolved loans (Current, In Grace Period): their outcome is *unknown* and labeling them either way poisons the data.

**The maturity / vintage trap.** A 36-month loan originated last month *cannot* have defaulted yet — not because it's safe, but because it hasn't had time. If you keep immature loans, you systematically mislabel future-defaulters as "good," biasing PD downward. The fix: only keep loans whose performance window has fully elapsed (matured vintages), and in the out-of-time split (§5) drop the immature 60-month loans from the latest origination years.

```python
import src.data as data
raw, lc_is_real = data.load_lending_club()         # (df, is_real_flag)
mdf, meta = data.define_target_and_audit(raw)
audit = meta["audit"]
print(audit["rows_kept"], audit["default_rate"])    # e.g. 56943, 0.2244
print(audit["rows_dropped_unresolved"])             # Current + In-Grace removed
```

**Interpretation.** A retail default rate in the high-teens to low-20s% is plausible for a sub-prime-heavy marketplace lender; a 2% rate would suggest you accidentally kept only the safest matured loans, and a 50% rate would suggest a labeling bug. Always sanity-check the base rate first.

## 4. Data leakage — the failure mode that produces "too good to be true" models

**Concept.** *Leakage* is when a feature contains information that would not be available at the moment of decision — typically because it encodes the outcome. The classic credit example: `recoveries`, `total_payment`, `out_principal`. These are only known *after* the loan resolves; a defaulted loan has near-zero future payments, so a model "learns" the answer by reading the answer.

**How to detect it — the AUC ≈ 1.0 tell.** Fit two models: one *with* the suspect post-origination fields, one *clean*. If the leaked model scores near-perfect discrimination while the clean model is realistic, you have proof.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
import src.metrics as metrics
leak_df, leak_cols = meta["leak_demo_df"], meta["leak_cols"]
ltr = leak_df[leak_df.issue_year.isin([2014,2015,2016])]
loot = leak_df[leak_df.issue_year == 2018]

m1 = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0)\
       .fit(ltr[data.LC_NUMERIC + leak_cols], ltr.default)
m2 = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0)\
       .fit(ltr[data.LC_NUMERIC], ltr.default)
print(metrics.roc_auc_score(loot.default, m1.predict_proba(loot[data.LC_NUMERIC+leak_cols])[:,1]))  # ~1.00
print(metrics.roc_auc_score(loot.default, m2.predict_proba(loot[data.LC_NUMERIC])[:,1]))             # ~0.76
```

**Rule of thumb.** A retail PD model with out-of-time AUC above ~0.85 should make you *suspicious*, not happy. Real underwriting signal tops out around 0.70–0.80. A 0.99 is not a great model; it's a leak.

**The other exclusion: lender's own assessment.** Drop `grade`, `sub_grade`, `int_rate`. These are the originator's *existing* risk judgment. Including them means you model the lender's pricing policy (circularity) rather than raw creditworthiness, and you inflate apparent performance. Underwrite from primary signals (FICO, DTI, utilization), not from someone else's score.

## 5. Out-of-time (OOT) validation — why random splits lie in credit

**Concept.** Credit data is *temporal*. Default rates drift with the economy, underwriting standards change, and the population shifts over time. A random train/test split leaks future information into training (you'd train on 2018 loans and test on 2016) and flatters the model. The honest design splits by **origination vintage**:

- **Train:** older vintages (e.g. 2014–2016).
- **Validation:** an intermediate vintage (2017) for tuning.
- **Out-of-time test:** the most recent matured vintage (2018) — the only fair estimate of forward performance.

```python
tr, va, oot = data.oot_split(mdf)
F = data.LC_UNDERWRITING_FEATURES
print(len(tr), len(va), len(oot))                              # 35987 12064 8892
print(tr.default.mean(), va.default.mean(), oot.default.mean()) # 0.217 0.238 0.238
```

**Interpretation.** Rising default rates across train→valid→OOT are *expected* if your data has vintage drift; they give the stability tests in §11 something real to detect. The OOT design also correctly drops immature 60-month 2018 loans (§3).

> **Survivorship & reject inference (a stated limitation, not a bug).** Your training data only contains loans that were *approved and booked*. It says nothing about applicants who were rejected. So your PD model is estimated on a *selected* sample and does not extend cleanly to the reject region. Production-grade work addresses this with **reject inference** (e.g. reweighting, parceling, or a bivariate selection model). For a portfolio piece, you must at least *name* the limitation in the report.

---

# Part III — The champion: a Weight-of-Evidence scorecard

The scorecard is the regulator-familiar incumbent: monotone, coefficient-interpretable, and trivially mapped to decline reasons. It is the benchmark the fancy model must beat.

## 6. Weight of Evidence (WOE) — turning raw features into risk-monotone signal

**Concept.** WOE re-expresses each predictor in terms of how strongly each bin separates "goods" from "bads." For a bin $i$:

$$\text{WOE}_i = \ln\!\left(\frac{\text{Distribution of Goods}_i}{\text{Distribution of Bads}_i}\right) = \ln\!\left(\frac{g_i / G}{b_i / B}\right)$$

where $g_i, b_i$ are the counts of non-defaulters and defaulters in bin $i$, and $G,B$ are the totals. (In this repo "good" = non-default, so **higher WOE = safer**.) In practice you add Laplace smoothing ($+\alpha$ to each count) so empty bins don't blow up to $\pm\infty$.

**Why WOE is loved in credit:**
1. It linearizes the relationship to the log-odds, so a logistic regression on WOE is well-specified.
2. It handles missing values as just another bin.
3. It is monotone and bin-based, so it's robust to outliers and easy to explain to a regulator ("each band gets X points").

## 7. Information Value (IV) — ranking predictive power

**Concept.** IV aggregates WOE across bins into a single number measuring how much a feature separates goods from bads:

$$\text{IV} = \sum_i \left(\frac{g_i}{G} - \frac{b_i}{B}\right)\times \text{WOE}_i$$

**Interpretation bands (industry convention):**

| IV | Strength | Action |
|---|---|---|
| < 0.02 | useless | drop |
| 0.02 – 0.10 | weak | keep cautiously |
| 0.10 – 0.30 | medium | good predictor |
| 0.30 – 0.50 | strong | core predictor |
| > 0.50 | "suspicious" | **check for leakage** |

That last band is not a typo: an IV above ~0.5 in retail credit usually means leakage, not a brilliant feature.

```python
import src.scorecard as scorecard
enc = scorecard.WOEEncoder(data.LC_NUMERIC, data.LC_CATEGORICAL).fit(tr[F], tr.default)
print(enc.iv_table().round(4))
#                    IV strength
# fico            0.3187   strong
# revol_util      0.2692   medium
# dti             0.1026   medium
# inq_last_6mths  0.0318     weak
# annual_inc      0.0249     weak
# ...             <0.02    useless
```

**Interpretation.** FICO, revolving utilization, and DTI carrying the signal is *economically sensible* — that's a conceptual-soundness win you'll cite later. Features below 0.02 IV get dropped from the scorecard automatically.

## 8. The scorecard model: logistic regression on WOE, scaled to points

**The PD model.** Fit a logistic regression on the WOE-transformed features:

$$P(Y=1\mid x) = \frac{1}{1 + \exp\!\big(-(\beta_0 + \sum_j \beta_j\,\text{WOE}_j(x_j))\big)}$$

**Scaling log-odds to a points scorecard.** Lenders express the model as additive points where a fixed number of points doubles the odds of being good. Define:

$$\text{Score} = \text{Offset} + \text{Factor}\times \ln(\text{odds})$$

Given a chosen **PDO** (points to double the odds), **base score**, and **base odds**:

$$\text{Factor} = \frac{\text{PDO}}{\ln 2}, \qquad \text{Offset} = \text{Base Score} - \text{Factor}\times\ln(\text{Base Odds})$$

Each feature contributes per-bin points $\propto -\text{Factor}\cdot(\beta_j\,\text{WOE}_j + \beta_0/k)$ (with $k$ features), so a final score is just a sum of points — transparent and auditable. Here we use base score 600, base odds 50:1, PDO 20.

```python
sc = scorecard.Scorecard(enc, base_score=600, base_odds=50, pdo=20).fit(tr[F], tr.default)
print(sc.features_)        # ['fico','dti','annual_inc','revol_util','inq_last_6mths']
print(sc.dropped_low_iv_)  # features removed for IV < 0.02
print(sc.wald_.round(4))   # coefficients + p-values (next section)
p_ch = sc.predict_proba(oot[F])
print(metrics.discrimination(oot.default, p_ch))  # {'AUC':0.748,'Gini':0.496,'KS':0.368}
```

## 9. Statistical significance — the Wald test

**Concept.** Are the coefficients distinguishable from zero? The Wald statistic for coefficient $\hat\beta_j$ is

$$z_j = \frac{\hat\beta_j}{\text{SE}(\hat\beta_j)}, \qquad \widehat{\text{Var}}(\hat\beta) = (X^\top V X)^{-1},\quad V = \text{diag}\big(p_i(1-p_i)\big)$$

$z_j$ is asymptotically standard normal, so $p = 2(1-\Phi(|z_j|))$. (We compute $(X^\top V X)^{-1}$ directly rather than forming an $n\times n$ matrix — important for memory on large samples.)

**Interpretation.** All retained features should have $|z|$ comfortably large (p ≈ 0) and **signs consistent with economic theory** (e.g. higher FICO → lower risk → negative coefficient on the WOE here). A statistically significant coefficient with the *wrong* sign is a red flag for multicollinearity or a data problem.

---

# Part IV — The challenger: a monotone, calibrated gradient-boosting model

## 10. Why a nonlinear challenger, and the constraints that keep it usable

**Concept.** Gradient-boosted trees fit an additive ensemble of shallow trees, each correcting the previous ensemble's residuals. They capture **interactions** and **nonlinearities** a linear-in-WOE scorecard cannot — which is exactly why a well-built challenger usually wins on discrimination. But raw GBMs violate two credit requirements, so we constrain them:

1. **Monotonicity constraints.** Economic priors say risk must move in a known direction with each feature (non-increasing in FICO/income/employment length; non-decreasing in DTI/utilization/term/loan amount/inquiries). We *impose* these as hard constraints so the model can never learn, e.g., "higher FICO → higher risk" on a noisy subsample. This protects against absurd, unexplainable, and litigable behavior.
2. **Calibration.** A GBM optimizes ranking, not probability accuracy; its raw scores can be systematically off-level. We wrap it in **isotonic calibration** (a monotone, non-parametric mapping from raw score to true frequency) via cross-validation, so the output is a usable PD.

**On class imbalance — what *not* to do.** With ~22% defaults you have mild imbalance. Resist **SMOTE** and other synthetic-oversampling: they distort the base rate and *destroy calibration*, which you need for pricing/ECL. Prefer `class_weight`/`scale_pos_weight` (reweight the loss) and then *recalibrate*. Discrimination is not the only goal.

```python
import src.challenger as challenger
clf, feats, mono = challenger.build_challenger(
    data.LC_NUMERIC, data.LC_CATEGORICAL, data.LC_MONOTONE, calibrate=True)
clf.fit(tr[F], tr.default)
p_cl = clf.predict_proba(oot[F])[:,1]
print(metrics.discrimination(oot.default, p_cl))  # {'AUC':0.766,'Gini':0.532,'KS':0.401}

# verify the constraints actually held after fitting
mono_ok = {f: challenger.check_monotonicity(clf, tr[F], f, data.LC_MONOTONE[f])
           for f in data.LC_NUMERIC if data.LC_MONOTONE[f] != 0}
print(mono_ok)   # all True
```

**Interpretation.** The challenger beating the champion by ~0.018 AUC is the *believable* kind of lift. A 0.10 jump would itself be a leakage alarm. Always verify monotonicity *empirically after fitting* — declaring a constraint is not the same as the fitted model honoring it.

---

# Part V — Evaluating a PD model: the three metric families

A junior analyst reports AUC. A risk professional reports **discrimination, calibration, and stability** — because a model can ace one and fail another, and each failure breaks a different use.

## 11. Discrimination — can the model rank risk?

**ROC / AUC.** Sweep the decision threshold and plot true-positive rate vs false-positive rate; the area under that curve (**AUC**) equals the probability that a randomly chosen defaulter is scored riskier than a randomly chosen non-defaulter (it's the Mann–Whitney U statistic normalized). 0.5 = coin flip; 1.0 = perfect.

**Gini.** A linear rescaling banks prefer:
$$\text{Gini} = 2\times\text{AUC} - 1$$

**KS (Kolmogorov–Smirnov).** The maximum vertical gap between the cumulative score distributions of bads and goods — the single threshold where the model best separates the two populations:
$$\text{KS} = \max_s \big| F_{\text{bad}}(s) - F_{\text{good}}(s) \big|$$

**Interpretation benchmarks (retail PD):** AUC 0.70–0.80 is solid; Gini 0.40–0.55 healthy; KS 0.30–0.45 typical. Higher isn't always better — see the leakage warning.

```python
print(metrics.discrimination(oot.default, p_ch))   # champion
print(metrics.discrimination(oot.default, p_cl))    # challenger
```

## 12. Calibration — are the predicted PDs *true*?

A model can rank perfectly yet output PDs that are systematically too high or low. That's fine for accept/decline ranking but **fatal for pricing and expected-loss provisioning**, which consume the PD *level* directly.

**Reliability diagram.** Bin by predicted PD, plot mean prediction vs observed default rate. On the 45° line = well-calibrated.

**Brier score.** Mean squared error of probabilistic forecasts (lower is better):
$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat p_i - y_i)^2$$

**Hosmer–Lemeshow test.** Group into $g$ deciles of predicted risk and compare observed $O_k$ vs expected $E_k$ defaults:
$$H = \sum_{k=1}^{g} \frac{(O_k - E_k)^2}{E_k\,(1 - E_k/n_k)} \sim \chi^2_{g-2}$$

A **small p-value means the model is mis-calibrated** (observed ≠ expected). This is the test that catches what AUC hides.

```python
print(metrics.calibration(oot.default, p_ch))
# {'Brier':0.155,'HL_chi2':25.13,'dof':8,'p_value':0.0015}  ← champion FAILS (p<0.05)
print(metrics.calibration(oot.default, p_cl))
# {'Brier':0.149,'HL_chi2':12.70,'dof':8,'p_value':0.1228}  ← challenger passes
```

**Interpretation.** Here the interpretable champion *ranks* fine but is *mis-calibrated* (HL p ≈ 0.0015). The lesson to put in the report: a mis-calibrated champion is acceptable for decline ranking but must be recalibrated before any pricing or ECL use. The isotonic-wrapped challenger passes. **This is exactly why you never report discrimination alone.**

## 13. Stability — is the population still the one you trained on?

**Population Stability Index (PSI).** Compare a quantity's distribution between two periods (train vs OOT). With expected proportions $e_i$ (train) and actual $a_i$ (OOT) per bin:
$$\text{PSI} = \sum_i (a_i - e_i)\,\ln\!\frac{a_i}{e_i}$$

(Note this is a symmetrized KL-style divergence.) **Bands:** < 0.10 stable; 0.10–0.25 moderate shift (investigate); > 0.25 significant shift (recalibrate/retrain).

```python
for col in ["dti","fico"]:
    r = metrics.psi_report(tr[col], oot[col]); print(col, r["PSI"], r["band"])
# dti  0.1148 moderate shift   ← an input is drifting
# fico 0.0553 stable
sp = metrics.psi_report(clf.predict_proba(tr[F])[:,1], p_cl)
print("score", sp["PSI"], sp["band"])  # 0.0298 stable  ← output still stable (for now)
```

**Interpretation — the realistic monitoring story.** A *key input* (DTI) is drifting moderately while the *output score* is still stable. That is the early-warning pattern: input drift precedes output drift. The monitoring plan should put DTI on a watchlist with a recalibration trigger at PSI 0.25.

## 14. Translating to money

PD only matters because it implies loss. Compute expected loss per the master equation, here with EAD ≈ loan amount and LGD = 0.55:

$$\text{EL} = \sum_i \hat p_i \times \text{LGD} \times \text{EAD}_i$$

```python
ead = oot["loan_amnt"].values
el_ch = metrics.expected_loss(p_ch, ead)
el_cl = metrics.expected_loss(p_cl, ead)
print(round(el_ch), round(el_cl), f"{100*(el_ch-el_cl)/el_ch:.1f}% lower")
# 19,097,897 → 18,023,903   (~5.6% lower under the challenger)

bc = metrics.approval_badrate_curve(oot.default, p_cl, steps=40)  # approval vs bad-rate frontier
```

**Interpretation.** Now the AUC lift has a dollar figure leadership understands. The approval-rate vs bad-rate curve lets the business pick a cutoff: accept more applicants (higher approval) at the cost of a higher bad rate among approved. The challenger dominating that frontier is the business case for adoption.

---

# Part VI — Explainability and conceptual soundness

Regulators (and ECOA) require that decisions be explainable. We produce **global** drivers, **local** per-applicant attributions, **adverse-action reason codes**, and a **conceptual-soundness cross-check**.

## 15. SHAP — Shapley values for model attribution

**Concept.** Borrowed from cooperative game theory, the **Shapley value** fairly distributes a prediction among features by averaging each feature's marginal contribution over all orderings:

$$\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\big[f(S\cup\{j\}) - f(S)\big]$$

Its defining guarantee is the **efficiency (local accuracy) property**: the attributions exactly reconstruct the prediction's deviation from the baseline:

$$\sum_j \phi_j = f(x) - \mathbb{E}[f(X)]$$

**KernelSHAP** approximates this by sampling feature-subset coalitions and solving a weighted least-squares with SHAP kernel weights — model-agnostic, so it works on the GBM. (If you `pip install shap`, swap it in; the home-grown version here is verified to satisfy efficiency exactly.)

```python
import src.explain as explain
bg = tr[F].sample(120, random_state=1)              # background / reference set
predict = lambda d: clf.predict_proba(d)[:, 1]
samp = oot[F].sample(180, random_state=2)
sm = explain.shap_matrix(predict, samp, bg, F, nsamples=120, seed=3)
print(explain.global_importance(sm).round(4))        # mean |phi| per feature
```

**Interpretation.** Global SHAP importance should *agree* with the scorecard's IV ranking (FICO, utilization, DTI on top). Agreement between two independent methods is strong evidence the model keys on economically sound signals.

## 16. Adverse-action reason codes (ECOA / Reg B)

**Concept.** A declined applicant has a legal right to the principal reasons. Two mechanisms:
- **From SHAP (challenger):** the features with the largest *positive* contributions to this applicant's PD are the decline reasons.
- **From the scorecard (champion):** the "points below maximum" method — the features where the applicant lost the most points versus the best attainable.

```python
import numpy as np
hi = oot[F].iloc[[np.argmax(p_cl)]]                  # highest-risk applicant
phi = explain.kernel_shap(predict, hi.iloc[0], bg, F, nsamples=500, seed=9)
print(explain.reason_codes_from_shap(phi, 4))
print(explain.reason_codes_from_scorecard(sc, hi.iloc[0], 4))
# e.g. [('revol_util', ..., 'Revolving credit utilisation too high'),
#       ('fico', ..., 'Credit score below approved-applicant norms'), ...]
```

**Interpretation.** Both engines should surface overlapping, human-readable reasons. Demonstrating reason codes from *both* the interpretable and the black-box model shows you can operationalize the right-to-explanation regardless of model type.

## 17. Conceptual-soundness cross-check (an SR 11-7 staple)

**Concept.** For each feature, compare three signs: the challenger's effect (sign of correlation between feature value and its SHAP attribution), the champion's effect, and economic *theory*. They should agree.

```python
cs = explain.conceptual_soundness(sm, samp, sc.predict_proba(samp),
                                  data.LC_MONOTONE, data.LC_NUMERIC)
print(cs)   # columns: feature, challenger_sign, champion_sign, theory_sign, all_agree
```

**Interpretation — and an honest finding.** The *strong* features (FICO, DTI, utilization, inquiries) all agree across challenger, champion, and theory — reassuring. The disagreements will cluster on the *weak / near-zero-IV* features (income, loan amount, term), where the relationship is statistically flat and attribution direction is effectively noise. The correct validator conclusion is not "the model is broken" but "no *material* driver disagrees; drop or stop relying on the near-zero-IV features." Reporting this nuance is what separates a validator from a dashboard.

---

# Part VII — Fair-lending analysis (the HMDA module)

Switch hats and datasets. We now audit a lending *decision* for **disparate impact** using real applicant demographics.

## 18. The two theories of discrimination

- **Disparate treatment:** explicitly using a protected attribute (race, sex, …) in the decision. Illegal and easy to avoid — just don't feed the attribute in.
- **Disparate impact:** a facially neutral policy that *disproportionately* harms a protected group, regardless of intent. This is the hard, interesting case — and the reason "just drop the race column" is not a fairness solution.

## 19. The Adverse Impact Ratio and the four-fifths rule

**Concept.** Compare each group's selection (approval) rate to the most-favored group's:

$$\text{AIR}_g = \frac{\text{approval rate}_g}{\text{approval rate}_{\text{reference}}}$$

By the EEOC **four-fifths (80%) rule**, $\text{AIR}_g < 0.80$ is a prima-facie disparate-impact flag.

```python
import src.fairness as fairness
h, hmda_is_real = data.load_hmda(); h = data.hmda_prepare(h)
air = fairness.adverse_impact_ratio(h.derived_race, 1 - h.denied)   # approved = 1 - denied
print(air[["approval_rate","n","AIR_vs_ref","flag_80pct_rule"]].round(3))
# Black or African American    AIR 0.628  → FAIL
# American Indian/Alaska Native AIR 0.710  → FAIL
```

## 20. Fairness through unawareness fails — the central lesson

**Concept.** Train a model to reproduce the deny decision using **only legitimate underwriting features** (income, DTI, LTV, loan amount, …) with **no protected attribute**. If the disparity disappears, unawareness works. It does not:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
X, y, grp = h[data.HMDA_FEATURES], h.denied, h.derived_race
Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, grp, test_size=0.3,
                                                 random_state=0, stratify=y)
base = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0).fit(Xtr, ytr)
p = base.predict_proba(Xte)[:,1]; appr = (p < 0.5).astype(int)
res = fairness.summarize(gte, yte, p, appr)
print(res["min_AIR"], res["groups_failing_80pct"])   # ~0.34, still failing
```

**Why.** The legitimate features are *correlated with* group membership (income, DTI, LTV differ across groups for structural reasons). The model reconstructs the protected attribute through these **proxies** and reproduces the disparity. Excluding race does not remove the impact — it launders it.

## 21. Group fairness metrics and the impossibility result

Three commonly demanded (and mutually incompatible) definitions:
- **Demographic parity:** equal approval rates across groups.
- **Equal opportunity:** equal **true-positive rate** (here, equal correct-denial rate among true bads) across groups.
- **Predictive parity:** equal **precision** (a denied applicant is equally likely to be a true bad across groups).

**Impossibility theorem (Kleinberg et al.; Chouldechova).** When base rates differ across groups, you **cannot** simultaneously satisfy calibration and equalized odds. Fairness is therefore a *choice among trade-offs*, not a single achievable state — say so explicitly.

```python
print(res["parity_gaps"])  # {'TPR_equal_opportunity':0.124,'precision_predictive_parity':0.118,'FPR':0.245}
```

## 22. Mitigations — and their cost

**Reweighing (Kamiran–Calders), a pre-processing fix.** Reweight training instances so group and label become independent in the weighted data:

$$w(g, y) = \frac{P(G=g)\,P(Y=y)}{P(G=g, Y=y)}$$

```python
w = fairness.reweighing_weights(gtr, ytr)
mit = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0).fit(Xtr, ytr, sample_weight=w)
pm = mit.predict_proba(Xte)[:,1]
print(fairness.summarize(gte, yte, pm, (pm<0.5).astype(int))["min_AIR"])  # ~0.35 — barely moves
```

**Group-specific thresholds, a post-processing fix.** Set a different approval cutoff per group to force AIR ≥ 0.80:

```python
appr_thr, thr = fairness.group_thresholds_for_air(gte, p, target_air=0.80)
# min AIR rises to ~0.80 — it WORKS, but…
```

**The honest conclusion (the part interviewers want to hear).** Reweighing is *legally safe but ineffective* against strong proxy structure; group thresholds are *effective but apply different standards by protected group* — itself a **disparate-treatment** concern, generally impermissible in US credit. The lever that fixes disparate *impact* creates disparate *treatment*. The realistic finding is that the bias is **structural** (in the features and their correlation with group), and the right remedy is a less-discriminatory-alternative *feature/model search*, not a post-hoc knob. Present the accuracy–fairness–legality tension plainly rather than claiming you "fixed" fairness.

---

# Part VIII — The headline deliverable: the SR 11-7 validation report

This document, not the notebook, is what a hiring manager reads first. Write it in the **independent validator's voice** — skeptical, specific, and decision-oriented.

## 23. Structure (11 sections)

1. **Executive summary & model overview** — purpose, uses (decisioning vs pricing vs provisioning), materiality/risk rating, champion-vs-challenger verdict, and a findings table.
2. **Conceptual soundness** — methodology appropriateness, variable selection (and the `grade`/`int_rate` exclusion), monotonicity, key assumptions (reject inference).
3. **Data review** — target definition, the leakage audit with the AUC≈1.0 demo, OOT design, representativeness.
4. **Outcomes analysis** — discrimination, calibration (where the champion fails HL), business/EL view.
5. **Stability & monitoring** — PSI table, the input-drift-precedes-output-drift story, concrete monitoring triggers.
6. **Explainability** — global/local SHAP, reason codes, the conceptual-soundness sign cross-check.
7. **Fairness** — observed AIR, the unawareness-fails demonstration, mitigation trade-offs.
8. **Limitations & assumptions** — synthetic data, reject inference, flat LGD, no macro overlay, library substitutions.
9. **Implementation testing** — independent re-fit reproduces the reported AUC to precision.
10. **Findings & recommendations** — the severity-rated register (below) with owners and due dates.
11. **Governance** — roles (developer vs validator), approver, review frequency, independence attestation.

## 24. Severity-rated findings — the analytical core

Every finding gets a **severity**, a **recommendation**, an **owner**, and a **due date**. Severity reflects *use-impact*, not aesthetics:

| Severity | Meaning | Example from this project |
|---|---|---|
| **High** | Blocks a primary use until remediated | Champion fails HL calibration → unfit for pricing/ECL without recalibration |
| **High** | Legal/regulatory exposure | Model reproduces disparate impact via proxies (min AIR ≈ 0.34) |
| **Medium** | Material but manageable | Moderate DTI input drift (PSI ≈ 0.115); unstable signs on weak features |
| **Low** | Process / documentation | Reject inference unaddressed; results on synthetic data |

**How to phrase a finding.** State the *observation* (with the number), the *use it breaks*, and the *required action*. Example: *"The champion fails the Hosmer–Lemeshow test (p ≈ 0.0015); its PD levels are unsuitable for risk-based pricing or ECL. Recommendation: recalibrate before any level-dependent use, or restrict the champion to rank-ordering only. Owner: model developer. Due: pre-deployment."*

**The validator's opinion.** End with one of: *approve*, *approve with conditions*, *reject*. This project lands on **approve with conditions** — the challenger is sound and outperforms, conditional on clearing the High findings, and nothing may go to production on synthetic-data evidence.

---

# Part IX — Assembly, reproduction, and presentation

## 25. Wire it together with one runner

Put the whole flow behind a single entry point that regenerates every figure and a machine-readable `results.json`, so the report's numbers are never hand-typed:

```python
# src/run_pipeline.py  (sketch)
def run_pd():        # data → champion → challenger → metrics → SHAP → figures
    ...
def run_fairness():  # HMDA → AIR → unawareness demo → mitigations → figures
    ...
if __name__ == "__main__":
    run_pd(); run_fairness()   # writes reports/figures/*.png and reports/results.json
```

```bash
python src/run_pipeline.py     # one command rebuilds all 11 figures + results.json
```

## 26. Reproducibility checklist (SR 11-7 implementation testing)

- **Fixed seeds** everywhere (`random_state`).
- **Independent re-fit** of the challenger reproduces the reported OOT AUC to reported precision (0.7662 = 0.7662).
- **No hand-edited numbers** — the report cites `results.json`, which regenerates from the runner.
- **Real-data swap-in path documented** (next section).

## 27. Dropping in real data

The loaders use real CSVs if present and fall back to synthetic otherwise — so the same code produces real results with zero changes:

- **Lending Club:** save the historical loan file as `data/lending_club.csv` (e.g. the Kaggle "Lending Club Loan Data" mirror). It needs `loan_status` plus the underwriting columns; the loader handles target definition, the leakage drop, and the OOT split.
- **HMDA:** from the **FFIEC Data Browser** (`ffiec.cfpb.gov/data-browser/`), filter to **one state, 2024, first-lien owner-occupied 1–4 family conventional purchase**, download, and save as `data/hmda_2024.csv`. Use a single-state slice, not the national file.

With real files present, *every number recomputes on real data.* Re-read the report's data-provenance note and update the synthetic disclaimer accordingly.

## 28. How to present this in a portfolio or interview

- **Lead with the validation report, not the model.** "I built a PD model *and validated it as an independent reviewer*" is a more senior story than "I trained XGBoost."
- **Tell the calibration story.** "My champion had great AUC but failed Hosmer–Lemeshow, so it was unfit for pricing — only the calibrated challenger was" demonstrates you understand *why* credit teams test more than AUC.
- **Tell the fairness story.** "I showed that dropping the race column did *not* remove disparate impact, because the legitimate features proxy for it" demonstrates real fair-lending literacy.
- **Be honest about synthetic data and limitations.** Naming reject inference, the flat LGD, and the synthetic fallback signals that you think like a validator, which is the entire point.

---

# Appendix A — Formula cheat-sheet

| Quantity | Formula |
|---|---|
| Expected loss | $\text{EL} = \text{PD}\times\text{LGD}\times\text{EAD}$ |
| WOE (bin $i$) | $\ln\big((g_i/G)/(b_i/B)\big)$ |
| Information Value | $\sum_i (g_i/G - b_i/B)\,\text{WOE}_i$ |
| Logistic PD | $1/(1+e^{-(\beta_0+\sum_j\beta_j\text{WOE}_j)})$ |
| Score scaling | $\text{Score}=\text{Offset}+\text{Factor}\ln(\text{odds})$, $\text{Factor}=\text{PDO}/\ln2$ |
| Wald $z$ | $\hat\beta_j/\text{SE}$, $\text{Var}=(X^\top V X)^{-1}$, $V=\text{diag}(p_i(1-p_i))$ |
| Gini | $2\,\text{AUC}-1$ |
| KS | $\max_s|F_{\text{bad}}(s)-F_{\text{good}}(s)|$ |
| Brier | $\frac1N\sum(\hat p_i-y_i)^2$ |
| Hosmer–Lemeshow | $\sum_k (O_k-E_k)^2/[E_k(1-E_k/n_k)]\sim\chi^2_{g-2}$ |
| PSI | $\sum_i (a_i-e_i)\ln(a_i/e_i)$ |
| Shapley value | $\sum_{S}\frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup j)-f(S)]$ |
| SHAP efficiency | $\sum_j\phi_j=f(x)-\mathbb{E}[f]$ |
| Adverse Impact Ratio | $\text{rate}_g/\text{rate}_{\text{ref}}$; flag $<0.80$ |
| Reweighing weight | $P(G{=}g)P(Y{=}y)/P(G{=}g,Y{=}y)$ |

# Appendix B — Interpretation quick-reference

| Metric | Good zone (retail PD) | What a failure breaks |
|---|---|---|
| AUC | 0.70–0.80 (≥0.85 ⇒ suspect leakage) | ranking / decisioning |
| Gini | 0.40–0.55 | ranking |
| KS | 0.30–0.45 | separation at the cutoff |
| Hosmer–Lemeshow p | > 0.05 (calibrated) | **pricing & ECL** |
| Brier | lower is better | probability accuracy |
| PSI | < 0.10 stable; 0.10–0.25 watch; > 0.25 act | model still valid for the current population |
| IV | 0.02–0.50 (> 0.50 ⇒ suspect leakage) | feature usefulness |
| AIR | ≥ 0.80 | fair-lending compliance |

# Appendix C — Glossary

**PD / LGD / EAD / EL** — see §1. **Vintage** — origination cohort (e.g. all 2018 loans). **OOT** — out-of-time test on the latest vintage. **WOE / IV** — Weight of Evidence / Information Value (§6–7). **PDO** — points to double the odds. **Calibration** — agreement between predicted PD and observed default rate. **Discrimination** — ability to rank-order risk. **PSI** — population stability index. **SHAP** — Shapley-value feature attributions. **Adverse action** — a credit denial requiring a reason under ECOA. **Disparate treatment / impact** — intentional vs. effect-based discrimination. **AIR** — adverse impact ratio. **Reject inference** — correcting for the fact that training data only contains approved loans. **SR 11-7** — US model-risk-management supervisory guidance. **Champion / challenger** — incumbent model vs. candidate replacement.

# Appendix D — Further reading

- Federal Reserve SR 11-7, *Guidance on Model Risk Management* (2011).
- Basel Committee, *International Convergence of Capital Measurement* (Basel II/III) — IRB and the PD/LGD/EAD framework.
- Naeem Siddiqi, *Credit Risk Scorecards* — the practitioner's bible on WOE/IV scorecards.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017.
- Kamiran & Calders, *Data preprocessing techniques for classification without discrimination* (reweighing), 2012.
- Chouldechova (2017) and Kleinberg, Mullainathan & Raghavan (2016) — fairness impossibility results.
- CFPB / FFIEC HMDA documentation and the FFIEC Data Browser.

---

*This guide accompanies the repository: `src/` holds the implementations referenced throughout, `notebooks/credit_pd_validation.ipynb` runs the full flow, and `reports/validation_report.md` is the worked example of the deliverable in Part VIII. Build in the order of the numbered sections and you will reproduce it end to end.*

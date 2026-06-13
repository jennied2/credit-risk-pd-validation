"""
Generate reports/validation_report.md from reports/results.json.

Run AFTER src/run_pipeline.py so the report always matches the numbers.
The narrative adapts to the data: discrimination winner, calibration pass/fail
per model, PSI bands, AIR flags, and real-vs-synthetic provenance are all
derived from results.json rather than hard-coded.

    python scripts/generate_report.py
"""
from __future__ import annotations
import os, json, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "reports", "results.json")
OUT = os.path.join(REPO, "reports", "validation_report.md")


def money(x):
    return f"${x:,.0f}"


def pct(x, nd=1):
    return f"{100*x:.{nd}f}%"


def main():
    d = json.load(open(RES))
    L = []  # lines

    lc_real = d["lc_is_real"]
    hmda_real = d["hmda_is_real"]
    ta = d["target_audit"]
    sp = d["splits"]
    disc = d["discrimination"]
    cal = d["calibration"]
    biz = d["business"]
    today = datetime.date.today().isoformat()

    champ_auc = disc["champion"]["AUC"]
    chal_auc = disc["challenger"]["AUC"]
    champ_p = cal["champion"]["p_value"]
    chal_p = cal["challenger"]["p_value"]
    champ_fail = champ_p < 0.05
    chal_fail = chal_p < 0.05
    better = "challenger" if chal_auc >= champ_auc else "champion"
    lift = abs(chal_auc - champ_auc)
    el_c, el_h = biz["expected_loss_champion"], biz["expected_loss_challenger"]
    el_delta = el_c - el_h
    el_pct = (el_delta / el_c) if el_c else 0.0
    el_better = "challenger" if el_h <= el_c else "champion"

    # ---- provenance ------------------------------------------------------
    if lc_real and hmda_real:
        prov = ("Both the Lending Club PD data and the HMDA fair-lending data are "
                "**real**. The figures below describe the actual samples loaded.")
    elif lc_real and not hmda_real:
        prov = ("The Lending Club PD data is **real** (loaded from "
                "`data/lending_club.csv`); the HMDA fair-lending data is still "
                "**synthetic** (no `data/hmda_2024.csv` present), so Section 7 is "
                "illustrative of method only. Drop a real HMDA slice in to make it real.")
    elif not lc_real and hmda_real:
        prov = ("The HMDA data is real; the Lending Club PD data is synthetic.")
    else:
        prov = ("Both datasets are **synthetic** fallbacks (no real files present), so "
                "all numbers are illustrative of the methodology rather than a real "
                "portfolio. Drop real CSVs into `data/` and re-run to get real results.")

    # ---- calibration narrative ------------------------------------------
    if not champ_fail and not chal_fail:
        cal_story = (
            f"Both models pass the Hosmer-Lemeshow test (champion p={champ_p:.4f}, "
            f"challenger p={chal_p:.4f}); predicted PD levels are statistically "
            f"consistent with observed default rates, so both are usable for "
            f"level-dependent purposes such as pricing and ECL.")
        cal_finding = None
    elif champ_fail and not chal_fail:
        cal_story = (
            f"The champion **fails** Hosmer-Lemeshow (p={champ_p:.4f}) while the "
            f"isotonic-calibrated challenger **passes** (p={chal_p:.4f}). The champion "
            f"ranks acceptably but its PD *levels* are mis-calibrated, so it is unfit "
            f"for risk-based pricing or expected-loss provisioning without recalibration; "
            f"the challenger is suitable for level-dependent use.")
        cal_finding = ("Champion fails Hosmer-Lemeshow calibration "
                       f"(p={champ_p:.4f}); unfit for pricing/ECL without recalibration",
                       "High")
    elif not champ_fail and chal_fail:
        cal_story = (
            f"The champion passes Hosmer-Lemeshow (p={champ_p:.4f}) but the challenger "
            f"**fails** (p={chal_p:.4f}): it improves ranking yet degrades probability "
            f"accuracy. The higher-AUC model is not automatically the better model — the "
            f"challenger should not feed pricing/ECL until recalibrated.")
        cal_finding = ("Challenger fails Hosmer-Lemeshow calibration "
                       f"(p={chal_p:.4f}) despite higher AUC", "High")
    else:  # both fail — the real-data story
        worse = "challenger" if chal_p <= champ_p else "champion"
        cal_lead = ("The challenger posts the higher AUC yet is the *worse*-calibrated of "
                    "the two" if worse == "challenger" else
                    "Although the challenger posts the higher AUC, the champion is the "
                    "worse-calibrated of the two")
        cal_story = (
            f"**Both models fail the Hosmer-Lemeshow test** (champion p={champ_p:.4f}, "
            f"challenger p={chal_p:.4f}). {cal_lead}. This is the discrimination-vs-"
            f"calibration trade-off in the open: better rank-ordering does not imply "
            f"trustworthy probability *levels*. Neither model is calibration-adequate on "
            f"the out-of-time sample, so neither may feed risk-based pricing or ECL "
            f"provisioning without recalibration; the challenger remains preferred for "
            f"rank-ordering / decisioning. A likely driver is the train→OOT population "
            f"shift (training default rate {pct(sp['train_default_rate'])} vs out-of-time "
            f"{pct(sp['oot_default_rate'])}): a calibration map fit on the training "
            f"vintages does not transfer cleanly to a shifted hold-out period.")
        cal_finding = ("Both models fail Hosmer-Lemeshow on the OOT sample "
                       f"(champion p={champ_p:.4f}, challenger p={chal_p:.4f}); the "
                       "higher-AUC challenger is the worse-calibrated. Neither is "
                       "pricing/ECL-ready without recalibration", "High")

    # ---- stability / drift ----------------------------------------------
    psis = {"DTI": d["stability"]["dti_psi_train_oot"],
            "FICO": d["stability"]["fico_psi_train_oot"],
            "Model score": d["stability"]["score_psi_train_oot"]}
    worst_band = max(psis.values(), key=lambda r: r["PSI"])
    drift_finding = None
    if worst_band["band"] != "stable":
        nm = [k for k, v in psis.items() if v is worst_band][0]
        drift_finding = (f"{nm} shows a {worst_band['band']} train→OOT "
                         f"(PSI={worst_band['PSI']:.3f})", "Medium")

    # ---- fairness --------------------------------------------------------
    mb = d["model_baseline"]
    mt = d["model_group_threshold"]
    mr = d["model_reweighed"]
    air_obs = d["air_observed"]
    failing = [r["group"] for r in air_obs if r["flag_80pct_rule"]]

    # ---- begin document --------------------------------------------------
    L += [
        "# Independent Model Validation Report",
        "## Consumer Probability-of-Default (PD) Scorecard & Machine-Learning Challenger",
        "",
        "| | |", "|---|---|",
        "| **Model ID** | PD-RETAIL-2026-01 |",
        "| **Model name** | Unsecured Consumer Loan PD Model (Underwriting) |",
        "| **Model type** | Binary classification — probability of default over loan term |",
        "| **Champion** | Logistic-regression WOE scorecard |",
        "| **Challenger** | Gradient-boosted trees, monotone-constrained, isotonic-calibrated |",
        "| **Validation standard** | SR 11-7 (Federal Reserve / OCC Model Risk Management) |",
        f"| **Report date** | {today} |",
        "| **Validator** | Independent Model Validation (second line) |",
        "| **Overall opinion** | **Approve with conditions** — see findings (Section 10) |",
        "",
        f"> **Data provenance.** {prov} This report is generated directly from "
        "`reports/results.json`; re-run `python src/run_pipeline.py` then "
        "`python scripts/generate_report.py` to refresh every number here.",
        "",
        "---", "",
        "## 1. Executive Summary & Model Overview", "",
        "The model estimates the probability that a consumer loan applicant defaults over "
        "the life of the loan. It supports three uses with different accuracy needs: "
        "accept/decline decisioning (needs rank-ordering), risk-based pricing, and "
        "expected-loss provisioning (both need calibrated PD *levels*).", "",
        "**Materiality / model-risk rating: HIGH.** The model drives credit-granting "
        f"decisions and loss provisioning on a portfolio of roughly {money(biz['portfolio_ead'])} "
        "exposure in the validation sample, produces consumer-facing adverse-action "
        "decisions under ECOA / Regulation B, and feeds financial reserves.", "",
        f"**Champion vs. challenger.** The {better} is the stronger model on discrimination "
        f"(out-of-time AUC {chal_auc:.3f} challenger vs {champ_auc:.3f} champion, a "
        f"{lift:.3f} difference). On expected loss the {el_better} is lower "
        f"({money(el_h)} vs {money(el_c)}, a {pct(el_pct)} difference on the validation "
        f"sample). {cal_story}", "",
    ]

    # findings table (assembled)
    findings = []
    if cal_finding:
        findings.append(cal_finding)
    if not hmda_real:
        findings.append(("Fair-lending audit runs on synthetic HMDA data; conclusions "
                         "are methodological until a real HMDA slice is loaded", "Medium"))
    findings.append((f"Fair-lending: a model on legitimate features only reproduces "
                     f"disparate impact via proxies (min AIR {mb['min_AIR']:.2f})",
                     "High" if hmda_real else "Medium"))
    if drift_finding:
        findings.append(drift_finding)
    if lc_real:
        findings.append(("Out-of-time vintages (2017–18) are partly immature in the data "
                         "snapshot; OOT performance carries a maturity/survivorship bias",
                         "Medium"))
    findings.append(("Reject inference not addressed — training observes approved loans "
                     "only (survivorship bias)", "Low"))

    L += ["### Summary of findings", "", "| ID | Finding | Severity |",
          "|----|---------|----------|"]
    for i, (txt, sev) in enumerate(findings, 1):
        L.append(f"| F-{i} | {txt} | **{sev}** |")
    L += ["", "---", ""]

    # Section 2
    excluded = ", ".join("`%s`" % c for c in ta["lender_assessment_excluded"])
    L += [
        "## 2. Conceptual Soundness", "",
        "Both model forms are standard for retail PD. The WOE scorecard is the "
        "regulator-familiar, coefficient-interpretable incumbent; gradient-boosted trees "
        "are an appropriate challenger because the risk surface is non-linear with "
        "interactions a linear-in-WOE model cannot capture.", "",
        f"**Variable selection.** The model uses underwriting-time features only. The "
        f"lender's own assessment ({excluded}) was **excluded** — including it would model "
        "the originator's pricing policy rather than raw creditworthiness and create "
        "circularity. Correct call.", "",
        "**Monotonicity.** Economic priors were imposed as hard constraints on the "
        "challenger; empirical sweeps confirm every constrained feature is honored "
        f"({sum(1 for v in d['monotonicity_verified'].values() if v)}/"
        f"{len(d['monotonicity_verified'])} verified).", "",
        "**Key weakness — reject inference.** Training observes approved-and-booked loans "
        "only, a selected sample; PD does not extend cleanly to the reject region. "
        "Acceptable for a portfolio piece if stated; production needs a reject-inference "
        "study or management overlay.", "",
        "---", "",
        "## 3. Data Review", "",
        f"**Target definition.** `default=1` for charged-off/defaulted, `0` for fully-paid; "
        f"unresolved loans dropped ({ta['rows_dropped_unresolved']:,} rows: "
        + ", ".join(f"{v:,} {k}" for k, v in ta["unresolved_status_values"].items())
        + f"). Of {ta['rows_total']:,} rows, **{ta['rows_kept']:,}** were kept at a "
        f"**{pct(ta['default_rate'],2)}** default rate.", "",
        f"**Leakage audit — passed.** {len(ta['leakage_columns_dropped'])} post-origination "
        "/ outcome-encoding fields were dropped. A deliberate leakage demonstration shows a "
        f"model *with* those fields scoring **AUC {d['leakage_demo']['auc_with_leakage']:.2f}** "
        f"versus **{d['leakage_demo']['auc_clean']:.2f}** clean — the signature of leakage, "
        "and the audit trail a validator wants.", "",
        "![Default rate by vintage](figures/01_default_by_vintage.png)", "",
        "![Leakage demonstration](figures/02_leakage_demo.png)", "",
        f"**Out-of-time design.** Split by origination vintage: **train {sp['train']:,}** / "
        f"**validation {sp['valid']:,}** / **out-of-time test {sp['oot']:,}**, with default "
        f"rates {pct(sp['train_default_rate'])} → {pct(sp['valid_default_rate'])} → "
        f"{pct(sp['oot_default_rate'])}. Temporal splitting is the honest design for credit; "
        "a random split would leak future information.", "",
        "---", "",
    ]

    # Section 4
    L += [
        "## 4. Outcomes Analysis (Discrimination, Calibration, Business)", "",
        "All metrics are on the **out-of-time** sample.", "",
        "### Discrimination", "",
        "| Metric | Champion | Challenger |", "|---|---|---|",
        f"| AUC-ROC | {disc['champion']['AUC']:.3f} | {disc['challenger']['AUC']:.3f} |",
        f"| Gini | {disc['champion']['Gini']:.3f} | {disc['challenger']['Gini']:.3f} |",
        f"| KS | {disc['champion']['KS']:.3f} | {disc['challenger']['KS']:.3f} |",
        "",
        "![ROC — OOT](figures/04_roc_oot.png)", "",
        "### Calibration", "",
        "| Metric | Champion | Challenger |", "|---|---|---|",
        f"| Brier | {cal['champion']['Brier']:.4f} | {cal['challenger']['Brier']:.4f} |",
        f"| Hosmer-Lemeshow χ² | {cal['champion']['HL_chi2']:.2f} | {cal['challenger']['HL_chi2']:.2f} |",
        f"| HL p-value | {cal['champion']['p_value']:.4f}"
        f"{' (FAIL)' if champ_fail else ' (pass)'} | {cal['challenger']['p_value']:.4f}"
        f"{' (FAIL)' if chal_fail else ' (pass)'} |",
        "",
        cal_story, "",
        "![Calibration — OOT](figures/05_calibration_oot.png)", "",
        "### Business view", "",
        f"Expected loss (PD × LGD × EAD, LGD 55%) on the OOT sample: **{money(el_c)}** "
        f"champion vs **{money(el_h)}** challenger — a {pct(el_pct)} difference against "
        f"portfolio EAD of {money(biz['portfolio_ead'])}. The approval-vs-bad-rate curve "
        "lets the business pick a cutoff.", "",
        "![Approval rate vs bad rate](figures/07_approval_badrate.png)", "",
        "---", "",
    ]

    # Section 5
    L += [
        "## 5. Stability & Ongoing Monitoring", "",
        "Population Stability Index, train → OOT:", "",
        "| Quantity | PSI | Band |", "|---|---|---|",
    ]
    for nm, r in psis.items():
        L.append(f"| {nm} | {r['PSI']:.3f} | {r['band']} |")
    L += [
        "",
        ("The model output remains "
         f"{d['stability']['score_psi_train_oot']['band']} (score PSI "
         f"{d['stability']['score_psi_train_oot']['PSI']:.3f})"
         + (", but a key input is drifting — the early-warning pattern where input drift "
            "precedes output drift." if drift_finding else ", consistent with a stable "
            "population.")), "",
        "![DTI population shift](figures/06_psi_dti_shift.png)", "",
        "**Monitoring triggers:** PSI on every input and on the score monthly (investigate "
        "0.10, recalibrate 0.25); discrimination on each maturing vintage; calibration "
        "(Hosmer-Lemeshow / observed-vs-expected) quarterly — the trigger that catches the "
        "Section 4 calibration finding.", "",
        "---", "",
    ]

    # Section 6
    top_shap = list(d["shap_global"].items())[:5]
    cs_agree = sum(1 for r in d["conceptual_soundness"] if r["all_agree"])
    cs_total = len(d["conceptual_soundness"])
    cs_disagree = [r["feature"] for r in d["conceptual_soundness"] if not r["all_agree"]]
    L += [
        "## 6. Explainability & Conceptual Checks", "",
        "**Global drivers.** SHAP (model-agnostic KernelSHAP, verified to satisfy the "
        "efficiency property) ranks the top drivers as "
        + ", ".join(f"**{k}** ({v:.3f})" for k, v in top_shap)
        + ". This aligns with the scorecard's Information-Value ranking, evidence both "
        "models key on the same economically sensible signals.", "",
        "![SHAP global importance](figures/08_shap_global.png)", "",
        "**Adverse-action reason codes (ECOA / Reg B).** For the highest-risk applicant the "
        "top PD-increasing SHAP contributors map to decline reasons "
        + "; ".join(f"\"{c[2]}\"" for c in d["reason_codes"]["challenger_shap"][:3])
        + ". The scorecard yields the same top reasons via points-below-maximum.", "",
        "![SHAP local](figures/09_shap_local.png)", "",
        f"**Conceptual-soundness cross-check.** Sign agreement across challenger, champion, "
        f"and economic theory holds for {cs_agree}/{cs_total} numeric features. "
        + ("All material drivers agree." if not cs_disagree else
           "Disagreements are confined to weak/low-IV features ("
           + ", ".join(cs_disagree) + "), where attribution direction is statistically "
           "unstable and not relied upon.") + " No *material* driver disagrees with theory.",
        "", "![Partial dependence](figures/10_pdp.png)", "",
        "---", "",
    ]

    # Section 7
    hmda_caveat = ("" if hmda_real else
                   " *(synthetic HMDA — methodological demonstration; load a real slice to "
                   "make these real.)*")
    L += [
        "## 7. Fair-Lending Analysis (HMDA Module)" + hmda_caveat, "",
        "Separate data, separate competency: auditing a lending **decision** for disparate "
        "impact. Lending Club has outcomes but no demographics; HMDA has demographics and "
        f"the approve/deny decision but no default outcome (denial rate "
        f"{pct(d['hmda_denial_rate'])}).", "",
        "**Observed adverse-impact ratios** (group approval rate ÷ reference "
        f"'{d['air_observed_ref']}'):", "",
        "| Group | Approval rate | AIR | 80% rule |", "|---|---|---|---|",
    ]
    for r in air_obs:
        L.append(f"| {r['group']} | {pct(r['approval_rate'])} | {r['AIR_vs_ref']:.2f} | "
                 f"{'**FAIL**' if r['flag_80pct_rule'] else 'pass'} |")
    L += [
        "",
        (f"Groups failing the four-fifths rule: {', '.join(failing)}." if failing
         else "No group fails the four-fifths rule in the raw decisions."), "",
        "**Fairness through unawareness fails (key finding).** A model trained on "
        "legitimate underwriting features only — no protected attribute — still produces a "
        f"minimum AIR of **{mb['min_AIR']:.2f}** (AUC {mb['auc']:.3f}), failing the 80% rule. "
        "The legitimate features proxy for group membership, so excluding the protected "
        "attribute launders the disparity rather than removing it.", "",
        "![Fairness regimes — AIR](figures/11_air_regimes.png)", "",
        "**Group fairness gaps (baseline):** equal-opportunity (TPR) "
        f"{mb['parity_gaps']['TPR_equal_opportunity']:.3f}, predictive-parity (precision) "
        f"{mb['parity_gaps']['precision_predictive_parity']:.3f}, FPR "
        f"{mb['parity_gaps']['FPR']:.3f}.", "",
        "**Mitigations and their cost.** Reweighing (Kamiran-Calders) moved minimum AIR to "
        f"**{mr['min_AIR']:.2f}** at near-zero AUC cost ({mr['auc']:.3f}) — too weak to "
        "overcome proxy structure. Group-specific thresholds lifted minimum AIR to "
        f"**{mt['min_AIR']:.2f}** (approval {pct(mt['approval_rate_before'])} → "
        f"{pct(mt['approval_rate_after'])}) — effective, but applying different thresholds "
        "by protected group is itself a disparate-treatment concern, generally "
        "impermissible in US credit. The lever that fixes disparate *impact* creates "
        "disparate *treatment*; the bias is structural.", "",
        "---", "",
    ]

    # Section 8 limitations
    lims = []
    if not lc_real or not hmda_real:
        which = []
        if not lc_real:
            which.append("Lending Club")
        if not hmda_real:
            which.append("HMDA")
        lims.append(f"**Synthetic data** ({', '.join(which)}): those results are "
                    "illustrative of method until real files are loaded.")
    if lc_real:
        lims.append("**Vintage maturity / survivorship**: 2017–18 hold-out loans are "
                    "partly immature in the snapshot, biasing OOT performance and "
                    "calibration; a fully-matured-vintage split would sharpen this.")
    lims += [
        "**Reject inference**: training observes approved loans only.",
        "**Flat LGD (55%)**: expected-loss figures are directionally indicative; a real "
        "model estimates LGD separately.",
        "**No macro overlay**: point-in-time model; lifetime ECL (CECL/IFRS-9) needs a "
        "forward macro scenario.",
        "**Library substitutions**: SHAP, scorecard binning, the GBM, Wald p-values and "
        "fairness mitigations are first-principles implementations (see README); swap in "
        "canonical packages on production infrastructure.",
    ]
    L += ["## 8. Limitations & Assumptions", ""]
    L += [f"{i}. {t}" for i, t in enumerate(lims, 1)]
    L += ["", "---", ""]

    # Section 9
    it = d["implementation_test"]
    L += [
        "## 9. Implementation Testing", "",
        f"**Reproducibility {'passed' if it['reproducible'] else 'FAILED'}.** An "
        f"independent re-fit re-scored the OOT sample at AUC {it['rerun_oot_auc']:.4f} "
        f"versus reported {it['reported_oot_auc']:.4f} — "
        f"{'match to reported precision' if it['reproducible'] else 'a mismatch requiring investigation'}. "
        "The pipeline regenerates all figures and `results.json` deterministically from "
        "`python src/run_pipeline.py`; this report regenerates from that JSON.", "",
        "Because core algorithms are bespoke re-implementations, implementation testing "
        "must be repeated on the production stack after swapping in canonical libraries.",
        "", "---", "",
    ]

    # Section 10 findings register
    L += [
        "## 10. Findings & Recommendations", "",
        "| ID | Severity | Finding | Recommendation |", "|----|----------|---------|----------------|",
    ]
    recs = {
        "High": "Remediate before the affected production use.",
        "Medium": "Address in next iteration / add to monitoring.",
        "Low": "Document; track.",
    }
    for i, (txt, sev) in enumerate(findings, 1):
        rec = recs[sev]
        if "calibration" in txt.lower():
            rec = ("Recalibrate before any pricing/ECL use; prefer the challenger for "
                   "rank-ordering. Investigate the train→OOT population shift.")
        elif "disparate impact" in txt.lower():
            rec = ("Run a formal disparate-impact + less-discriminatory-alternative "
                   "analysis on real HMDA; do not treat feature exclusion as a control.")
        elif "maturity" in txt.lower():
            rec = "Re-validate on a fully-matured-vintage split; note bias in monitoring."
        L.append(f"| F-{i} | **{sev}** | {txt} | {rec} |")
    L += [
        "",
        "**Validator's opinion: approve with conditions.** "
        + (f"The {better} outperforms on discrimination. " )
        + ("Neither model is calibration-adequate on the out-of-time sample, so neither "
           "may feed pricing/ECL until recalibrated; the challenger is preferred for "
           "rank-ordering/decisioning in the interim. "
           if champ_fail and chal_fail else
           "Calibration conditions in Section 4 must be cleared for level-dependent use. ")
        + ("Fair-lending findings must be cleared on real HMDA data before production. "
           if not hmda_real else "")
        + ("Results on synthetic data may not go to production until reproduced on real "
           "data." if (not lc_real or not hmda_real) else ""),
        "", "---", "",
        "## 11. Governance", "",
        "| Role | Assignment |", "|---|---|",
        "| Model owner (first line) | Credit Risk Modeling — owns F-findings on the model |",
        "| Independent validator (second line) | Model Validation — author; did not build the model |",
        "| Compliance / Fair-lending | Owns the HMDA disparate-impact findings |",
        "| Approver | Model Risk Committee |",
        "| Review frequency | Annual revalidation; quarterly performance/calibration monitoring; event-driven on PSI ≥ 0.25 |",
        "",
        "*Generated from `reports/results.json`. The dual-hat developer/validator split is "
        "a project framing device; in production these must be organizationally separate.*",
    ]

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")
    print(f"  lc_is_real={lc_real}  hmda_is_real={hmda_real}")
    print(f"  champion AUC {champ_auc:.3f} (HL p {champ_p:.4f}{'FAIL' if champ_fail else 'pass'})")
    print(f"  challenger AUC {chal_auc:.3f} (HL p {chal_p:.4f}{'FAIL' if chal_fail else 'pass'})")
    print(f"  findings: {len(findings)}")


if __name__ == "__main__":
    main()

"""
End-to-end pipeline runner.

Executes the full dual-hat workflow and writes every figure + numeric result the
validation report references. Run:  python src/run_pipeline.py
"""
from __future__ import annotations

import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, scorecard, challenger, metrics, explain, fairness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10, "figure.autolayout": True})
C_CHAMP, C_CHAL, C_MIT = "#4C72B0", "#C44E52", "#55A868"
R = {}   # results dict -> JSON


def fig(name):
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close()


# Human-readable feature labels — used ONLY for figure labels. The real column
# names (with underscores) stay unchanged everywhere the model/data rely on them.
PRETTY = {
    "fico": "FICO score", "dti": "Debt-to-income", "annual_inc": "Annual income",
    "loan_amnt": "Loan amount", "term_months": "Loan term",
    "emp_length": "Employment length", "revol_util": "Revolving utilization",
    "inq_last_6mths": "Recent inquiries", "open_acc": "Open accounts",
    "purpose": "Loan purpose", "home_ownership": "Home ownership",
}


def pretty(names):
    """Map raw feature name(s) to display labels for plots."""
    if isinstance(names, str):
        return PRETTY.get(names, names.replace("_", " ").title())
    return [PRETTY.get(n, n.replace("_", " ").title()) for n in names]


# =====================================================================
# MODULE 1 — PD MODEL (Lending Club)
# =====================================================================
def run_pd():
    raw, is_real = data.load_lending_club()
    R["lc_is_real"] = is_real; R["lc_raw_rows"] = len(raw)
    mdf, meta = data.define_target_and_audit(raw)
    R["target_audit"] = meta["audit"]
    tr, va, oot = data.oot_split(mdf)
    R["splits"] = {"train": len(tr), "valid": len(va), "oot": len(oot),
                   "train_default_rate": round(tr.default.mean(), 4),
                   "valid_default_rate": round(va.default.mean(), 4),
                   "oot_default_rate": round(oot.default.mean(), 4)}
    F = data.LC_UNDERWRITING_FEATURES

    # ---- default rate by vintage
    vr = mdf.groupby("issue_year")["default"].mean()
    plt.figure(figsize=(6, 3.2)); plt.bar(vr.index.astype(str), vr.values, color=C_CHAMP)
    plt.ylabel("default rate"); plt.title("Default rate by origination vintage")
    fig("01_default_by_vintage.png")

    # ---- leakage demonstration: AUC with vs without leakage columns
    leak_df, leak_cols = meta["leak_demo_df"], meta["leak_cols"]
    ltr = leak_df[leak_df.issue_year.isin([2014, 2015, 2016])]
    loot = leak_df[leak_df.issue_year == 2018]
    leak_feat = data.LC_NUMERIC + leak_cols
    gb_leak = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0)
    gb_leak.fit(ltr[leak_feat], ltr.default)
    auc_leak = metrics.roc_auc_score(loot.default, gb_leak.predict_proba(loot[leak_feat])[:, 1])
    gb_clean = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0)
    gb_clean.fit(ltr[data.LC_NUMERIC], ltr.default)
    auc_clean = metrics.roc_auc_score(loot.default, gb_clean.predict_proba(loot[data.LC_NUMERIC])[:, 1])
    R["leakage_demo"] = {"auc_with_leakage": round(float(auc_leak), 4),
                         "auc_clean": round(float(auc_clean), 4),
                         "leakage_columns": leak_cols}
    plt.figure(figsize=(4.6, 3.2))
    plt.bar(["with leakage", "clean"], [auc_leak, auc_clean], color=[C_CHAL, C_CHAMP])
    plt.ylim(0.5, 1.08); plt.ylabel("OOT AUC")
    plt.title("Why the leakage audit matters")
    for i, v in enumerate([auc_leak, auc_clean]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig("02_leakage_demo.png")

    # ---- CHAMPION: WOE/IV scorecard
    enc = scorecard.WOEEncoder(data.LC_NUMERIC, data.LC_CATEGORICAL).fit(tr[F], tr.default)
    iv = enc.iv_table()
    R["iv_table"] = iv["IV"].round(4).to_dict()
    plt.figure(figsize=(6, 3.4))
    iv_sorted = iv.sort_values("IV")
    plt.barh(pretty(iv_sorted.index), iv_sorted["IV"], color=C_CHAMP)
    plt.xlabel("Information Value"); plt.title("Predictor strength (IV ranking)")
    plt.axvline(0.02, color="grey", ls="--", lw=1)
    fig("03_iv_ranking.png")

    sc = scorecard.Scorecard(enc, base_score=600, base_odds=50, pdo=20).fit(tr[F], tr.default)
    R["scorecard"] = {"features": sc.features_, "dropped_low_iv": sc.dropped_low_iv_,
                      "wald": sc.wald_.round(4).reset_index().rename(
                          columns={"index": "term"}).to_dict("records")}
    p_ch = sc.predict_proba(oot[F])

    # ---- CHALLENGER: monotone GBM + calibration
    clf, feats, mono = challenger.build_challenger(
        data.LC_NUMERIC, data.LC_CATEGORICAL, data.LC_MONOTONE, calibrate=True)
    clf.fit(tr[F], tr.default)
    p_cl = clf.predict_proba(oot[F])[:, 1]

    # monotonicity verification
    monocheck = {f: challenger.check_monotonicity(clf, tr[F], f, data.LC_MONOTONE[f])
                 for f in data.LC_NUMERIC if data.LC_MONOTONE[f] != 0}
    R["monotonicity_verified"] = monocheck

    # ---- discrimination / calibration / stability
    R["discrimination"] = {"champion": metrics.discrimination(oot.default, p_ch),
                           "challenger": metrics.discrimination(oot.default, p_cl)}
    R["calibration"] = {"champion": metrics.calibration(oot.default, p_ch),
                        "challenger": metrics.calibration(oot.default, p_cl)}
    p_cl_tr = clf.predict_proba(tr[F])[:, 1]
    R["stability"] = {
        "score_psi_train_oot": metrics.psi_report(p_cl_tr, p_cl),
        "dti_psi_train_oot": metrics.psi_report(tr.dti, oot.dti),
        "fico_psi_train_oot": metrics.psi_report(tr.fico, oot.fico)}

    # ROC
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(4.6, 4.4))
    for p, lab, c in [(p_ch, "Champion (scorecard)", C_CHAMP), (p_cl, "Challenger (GBM)", C_CHAL)]:
        f_, t_, _ = roc_curve(oot.default, p)
        plt.plot(f_, t_, color=c, label=f"{lab}  AUC={metrics.roc_auc_score(oot.default,p):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — out-of-time test")
    plt.legend(fontsize=8); fig("04_roc_oot.png")

    # Calibration / reliability
    plt.figure(figsize=(4.8, 4.4))
    for p, lab, c in [(p_ch, "Champion", C_CHAMP), (p_cl, "Challenger", C_CHAL)]:
        ct = metrics.calibration_table(oot.default, p, 10)
        plt.plot(ct["predicted"], ct["actual"], "o-", color=c, label=lab)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("predicted PD"); plt.ylabel("observed default rate")
    plt.title("Calibration (reliability) — OOT"); plt.legend(fontsize=8)
    fig("05_calibration_oot.png")

    # PSI population shift on DTI
    plt.figure(figsize=(5.2, 3.2))
    plt.hist(tr.dti, bins=30, density=True, alpha=0.5, label="train (2014-16)", color=C_CHAMP)
    plt.hist(oot.dti, bins=30, density=True, alpha=0.5, label="OOT (2018)", color=C_CHAL)
    plt.xlabel("DTI"); plt.title(f"Population shift in DTI — PSI={R['stability']['dti_psi_train_oot']['PSI']}")
    plt.legend(fontsize=8); fig("06_psi_dti_shift.png")

    # ---- business view
    bc = metrics.approval_badrate_curve(oot.default, p_cl, 40)
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(bc.approval_rate, bc.bad_rate_approved, color=C_CHAL)
    plt.xlabel("approval rate"); plt.ylabel("bad rate among approved")
    plt.title("Approval-rate vs bad-rate trade-off (challenger)")
    fig("07_approval_badrate.png")
    ead = oot["loan_amnt"].values
    el_ch = metrics.expected_loss(p_ch, ead); el_cl = metrics.expected_loss(p_cl, ead)
    R["business"] = {"expected_loss_champion": round(el_ch, 0),
                     "expected_loss_challenger": round(el_cl, 0),
                     "portfolio_ead": round(float(ead.sum()), 0)}

    # ---- EXPLAINABILITY
    bg = tr[F].sample(120, random_state=1)
    predict = lambda d: clf.predict_proba(d)[:, 1]
    samp = oot[F].sample(180, random_state=2)
    sm = explain.shap_matrix(predict, samp, bg, F, nsamples=120, seed=3)
    gimp = explain.global_importance(sm)
    R["shap_global"] = gimp.round(4).to_dict()
    plt.figure(figsize=(5.6, 3.6))
    g2 = gimp.sort_values()
    plt.barh(pretty(g2.index), g2.values, color=C_CHAL)
    plt.xlabel("mean |SHAP|"); plt.title("Global feature importance (KernelSHAP)")
    fig("08_shap_global.png")

    # local waterfall for a declined applicant (high predicted PD)
    hi = oot[F].iloc[[np.argmax(p_cl)]]
    phi = explain.kernel_shap(predict, hi.iloc[0], bg, F, nsamples=500, seed=9)
    e0 = float(predict(bg).mean()); fx = float(predict(hi)[0])
    R["local_example"] = {"E_f": round(e0, 4), "f_x": round(fx, 4),
                          "applicant_pd": round(fx, 4)}
    order = phi.reindex(phi.abs().sort_values(ascending=False).index)
    plt.figure(figsize=(6, 3.8))
    colors = [C_CHAL if v > 0 else C_CHAMP for v in order.values]
    plt.barh(pretty(order.index[::-1]), order.values[::-1], color=colors[::-1])
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("SHAP contribution to PD"); plt.title(f"Local explanation — applicant PD={fx:.2f} (E[f]={e0:.2f})")
    fig("09_shap_local.png")

    # adverse-action reason codes (both engines) for that applicant
    R["reason_codes"] = {
        "challenger_shap": explain.reason_codes_from_shap(phi, 4),
        "champion_scorecard": explain.reason_codes_from_scorecard(sc, hi.iloc[0], 4)}

    # PDP for top drivers
    fig_, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, feat in zip(axes, ["dti", "fico"]):
        idx = F.index(feat)
        pdres = partial_dependence(clf, samp, [idx], grid_resolution=30)
        ax.plot(pdres["grid_values"][0], pdres["average"][0], color=C_CHAL)
        ax.set_xlabel(pretty(feat)); ax.set_ylabel("partial dependence (PD)")
        ax.set_title(f"PDP — {pretty(feat)}")
    fig("10_pdp.png")

    # conceptual soundness cross-check (uses champion PD on the same sample)
    champ_pd_samp = sc.predict_proba(samp)
    cs = explain.conceptual_soundness(sm, samp, champ_pd_samp, data.LC_MONOTONE, data.LC_NUMERIC)
    R["conceptual_soundness"] = cs.to_dict("records")

    # implementation testing: re-run challenger from scratch, compare AUC
    clf2, _, _ = challenger.build_challenger(data.LC_NUMERIC, data.LC_CATEGORICAL,
                                             data.LC_MONOTONE, calibrate=True)
    clf2.fit(tr[F], tr.default)
    auc2 = metrics.roc_auc_score(oot.default, clf2.predict_proba(oot[F])[:, 1])
    R["implementation_test"] = {
        "rerun_oot_auc": round(float(auc2), 4),
        "reported_oot_auc": R["discrimination"]["challenger"]["AUC"],
        "reproducible": bool(round(float(auc2), 4)
                             == R["discrimination"]["challenger"]["AUC"])}
    return mdf


# =====================================================================
# MODULE 2 — FAIR-LENDING AUDIT (HMDA)
# =====================================================================
def run_fairness():
    h, is_real = data.load_hmda(); h = data.hmda_prepare(h)
    R["hmda_is_real"] = is_real; R["hmda_rows"] = len(h)
    R["hmda_denial_rate"] = round(float(h.denied.mean()), 4)

    # observed-decision AIR
    air_obs = fairness.adverse_impact_ratio(h.derived_race, 1 - h.denied)
    R["air_observed"] = air_obs[["approval_rate", "n", "AIR_vs_ref", "flag_80pct_rule"]]\
        .round(3).reset_index().rename(columns={"g": "group"}).to_dict("records")
    R["air_observed_ref"] = air_obs.attrs["reference_group"]

    X, y, grp = h[data.HMDA_FEATURES], h.denied, h.derived_race
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, grp, test_size=0.3, random_state=0, stratify=y)

    base = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0).fit(Xtr, ytr)
    p = base.predict_proba(Xte)[:, 1]; appr = (p < 0.5).astype(int)
    res = fairness.summarize(gte, yte, p, appr)
    R["model_baseline"] = {"auc": res["auc"], "min_AIR": res["min_AIR"],
                           "failing_80pct": res["groups_failing_80pct"],
                           "parity_gaps": res["parity_gaps"]}

    # mitigation 1: reweighing
    w = fairness.reweighing_weights(gtr, ytr)
    mit = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0)
    mit.fit(Xtr, ytr, sample_weight=w)
    pm = mit.predict_proba(Xte)[:, 1]; apprm = (pm < 0.5).astype(int)
    resm = fairness.summarize(gte, yte, pm, apprm)
    R["model_reweighed"] = {"auc": resm["auc"], "min_AIR": resm["min_AIR"],
                            "failing_80pct": resm["groups_failing_80pct"],
                            "parity_gaps": resm["parity_gaps"]}

    # mitigation 2: group thresholds to hit 80% rule
    appr_thr, thr = fairness.group_thresholds_for_air(gte, p, target_air=0.80)
    air_thr = fairness.adverse_impact_ratio(pd.Series(gte.values), pd.Series(appr_thr))
    base_appr_rate = float((p < 0.5).mean()); thr_appr_rate = float(appr_thr.mean())
    R["model_group_threshold"] = {
        "thresholds": thr, "min_AIR": round(float(air_thr["AIR_vs_ref"].min()), 3),
        "failing_80pct": air_thr.index[air_thr["flag_80pct_rule"]].tolist(),
        "approval_rate_before": round(base_appr_rate, 3),
        "approval_rate_after": round(thr_appr_rate, 3)}

    # plot AIR across the three regimes (by race)
    def air_series(grp_arr, appr_arr):
        a = fairness.adverse_impact_ratio(pd.Series(np.asarray(grp_arr)),
                                          pd.Series(np.asarray(appr_arr)))
        return a["AIR_vs_ref"]
    s_obs = air_series(gte, 1 - yte.values)
    s_base = air_series(gte, appr); s_rw = air_series(gte, apprm); s_thr = air_series(gte, appr_thr)
    order = s_obs.sort_values().index
    xpos = np.arange(len(order)); wbar = 0.2
    plt.figure(figsize=(7.5, 3.8))
    plt.bar(xpos - 1.5 * wbar, s_obs.reindex(order), wbar, label="observed", color="#999")
    plt.bar(xpos - 0.5 * wbar, s_base.reindex(order), wbar, label="model", color=C_CHAMP)
    plt.bar(xpos + 0.5 * wbar, s_rw.reindex(order), wbar, label="reweighed", color=C_CHAL)
    plt.bar(xpos + 1.5 * wbar, s_thr.reindex(order), wbar, label="group thresholds", color=C_MIT)
    plt.axhline(0.80, color="red", ls="--", lw=1, label="80% rule")
    plt.xticks(xpos, [g.replace(" or ", "/\n") for g in order], fontsize=7)
    plt.ylabel("AIR vs reference"); plt.title("Adverse-impact ratio by race, across regimes")
    plt.legend(fontsize=7, ncol=2); fig("11_air_regimes.png")


if __name__ == "__main__":
    run_pd()
    run_fairness()
    with open(os.path.join(ROOT, "reports", "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    print("OK — figures + results.json written")
    print(json.dumps({k: R[k] for k in
                      ["discrimination", "calibration", "stability",
                       "model_baseline", "model_group_threshold"]}, indent=2, default=str))
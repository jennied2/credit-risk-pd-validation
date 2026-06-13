"""
Fair-lending audit (HMDA module).
 
HMDA has applicant race / ethnicity / sex and the approve-deny decision, but no
default outcome -- so it cannot train a PD model. Instead we model the *lending
decision* (denied = 1) on legitimate underwriting factors and audit it for
disparate impact. The observed decision serves as ground truth for the
error-rate parities.
 
IMPORTANT LIMITATION (state it -- maturity signal): this is a methodology
demonstration. Real fair-lending analysis controls for the full legitimate
credit profile and uses regulator-defined methodologies; group effects here may
be partly explained by unobserved factors.
 
Replaces fairlearn: metrics + Kamiran-Calders reweighing are implemented directly.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
 
 
# ---------------------------------------------------------------------------
# Group fairness metrics. Convention: positive (adverse) outcome = DENIED.
# ---------------------------------------------------------------------------
def selection_rates(group: pd.Series, approved: pd.Series) -> pd.DataFrame:
    """Approval (selection) rate per group -> basis for the 80% rule."""
    df = pd.DataFrame({"g": group.values, "approved": approved.values})
    t = df.groupby("g")["approved"].agg(["mean", "size"])
    t.columns = ["approval_rate", "n"]
    return t.sort_values("approval_rate", ascending=False)
 
 
def adverse_impact_ratio(group: pd.Series, approved: pd.Series,
                         reference: str | None = None) -> pd.DataFrame:
    """Adverse-impact ratio = group approval rate / reference approval rate.
    AIR < 0.80 flags potential disparate impact (the four-fifths rule)."""
    sr = selection_rates(group, approved)
    ref = reference if reference is not None else sr.index[0]   # highest-approval group
    sr["AIR_vs_ref"] = (sr["approval_rate"] / sr.loc[ref, "approval_rate"]).round(3)
    sr["flag_80pct_rule"] = sr["AIR_vs_ref"] < 0.80
    sr.attrs["reference_group"] = ref
    return sr
 
 
def _rate(mask_num, mask_den):
    d = mask_den.sum()
    return float(mask_num.sum() / d) if d else np.nan
 
 
def error_rate_parity(group, y_true, y_pred) -> pd.DataFrame:
    """Per-group TPR (equal opportunity) and precision (predictive parity),
    with y=1 = denied. TPR = of truly-denied, share predicted denied.
    Precision = of predicted-denied, share truly denied."""
    df = pd.DataFrame({"g": np.asarray(group), "y": np.asarray(y_true),
                       "p": np.asarray(y_pred)})
    rows = []
    for g, sub in df.groupby("g"):
        tp = ((sub.p == 1) & (sub.y == 1)).sum()
        fp = ((sub.p == 1) & (sub.y == 0)).sum()
        fn = ((sub.p == 0) & (sub.y == 1)).sum()
        tn = ((sub.p == 0) & (sub.y == 0)).sum()
        rows.append({"group": g, "n": len(sub),
                     "TPR_equal_opportunity": _rate(pd.Series([tp]), pd.Series([tp + fn])),
                     "precision_predictive_parity": _rate(pd.Series([tp]), pd.Series([tp + fp])),
                     "FPR": _rate(pd.Series([fp]), pd.Series([fp + tn]))})
    return pd.DataFrame(rows).set_index("group").round(3)
 
 
def parity_gaps(parity_tbl: pd.DataFrame) -> dict:
    """Max-min gap across groups for each parity metric (0 = perfect parity)."""
    return {c: round(float(parity_tbl[c].max() - parity_tbl[c].min()), 3)
            for c in ["TPR_equal_opportunity", "precision_predictive_parity", "FPR"]}
 
 
# ---------------------------------------------------------------------------
# Mitigation: Kamiran-Calders reweighing (pre-processing)
# ---------------------------------------------------------------------------
def reweighing_weights(group: pd.Series, label: pd.Series) -> np.ndarray:
    """Sample weights making group and label statistically independent in the
    training data: w(g,y) = P(g)P(y) / P(g,y)."""
    g = np.asarray(group); y = np.asarray(label)
    n = len(g)
    w = np.ones(n, dtype=float)
    pg = pd.Series(g).value_counts(normalize=True).to_dict()
    py = pd.Series(y).value_counts(normalize=True).to_dict()
    joint = pd.crosstab(pd.Series(g), pd.Series(y), normalize=True)
    for i in range(n):
        p_joint = joint.loc[g[i], y[i]]
        w[i] = (pg[g[i]] * py[y[i]]) / p_joint if p_joint > 0 else 1.0
    return w
 
 
def group_thresholds_for_air(group, pd_denied, target_air: float = 0.80,
                             ref_threshold: float = 0.50):
    """Post-processing lever: pick a per-group denial threshold so each group's
    approval rate >= target_air * reference-group approval rate.
 
    Returns (approved_pred array, thresholds dict). NOTE: this uses the
    protected attribute explicitly at decision time -> raises DISPARATE-TREATMENT
    concerns even as it fixes disparate impact. That legal tension is the point.
    """
    g = np.asarray(group); s = np.asarray(pd_denied)
    # Reference group = highest approval rate at the base threshold.
    appr_base = {grp: float((s[g == grp] < ref_threshold).mean())
                 for grp in pd.unique(g)}
    ref_rate = max(appr_base.values())
    target_rate = target_air * ref_rate
    thresholds, approved = {}, np.zeros(len(g), dtype=int)
    for grp in pd.unique(g):
        idx = g == grp
        if appr_base[grp] >= target_rate:
            thr = ref_threshold
        else:
            # Raise threshold (deny less) until the group hits target approval.
            thr = float(np.quantile(s[idx], target_rate))
        thresholds[grp] = round(thr, 3)
        approved[idx] = (s[idx] < thr).astype(int)
    return approved, thresholds
 
 
def summarize(group, y_true, y_pred_proba, approved_pred, ref=None) -> dict:
    """Bundle AIR + parities + AUC for a model's decisions."""
    air = adverse_impact_ratio(pd.Series(group), pd.Series(approved_pred), ref)
    par = error_rate_parity(group, y_true, 1 - np.asarray(approved_pred))
    return {
        "auc": round(float(roc_auc_score(y_true, y_pred_proba)), 4),
        "min_AIR": round(float(air["AIR_vs_ref"].min()), 3),
        "groups_failing_80pct": air.index[air["flag_80pct_rule"]].tolist(),
        "parity_gaps": parity_gaps(par),
        "air_table": air, "parity_table": par,
    }
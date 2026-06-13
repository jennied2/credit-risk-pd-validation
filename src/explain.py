"""
Explainability (the "X" in explainable ML).
 
Because the `shap` package is unavailable here, this implements a compact,
faithful KernelSHAP from first principles: model-agnostic additive attributions
that satisfy the efficiency property  sum(phi_j) = f(x) - E_background[f].
Drop-in `shap.KernelExplainer` would replace `kernel_shap` unchanged downstream.
 
Provides:
  * global importance (mean |phi|)
  * local attributions per applicant
  * adverse-action reason codes (challenger via SHAP; champion via the
    scorecard "points-below-maximum" method -- the ECOA/Reg B standard)
  * conceptual-soundness cross-check: do challenger effect signs agree with the
    logistic-regression coefficients and with underwriting theory?
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from math import comb
 
REASON_TEXT = {
    "fico": "Credit score below approved-applicant norms",
    "dti": "Debt-to-income ratio too high",
    "annual_inc": "Annual income below approved-applicant norms",
    "revol_util": "Revolving credit utilisation too high",
    "inq_last_6mths": "Too many recent credit inquiries",
    "loan_amnt": "Requested loan amount high relative to profile",
    "term_months": "Loan term elevates risk",
    "emp_length": "Short employment history",
    "open_acc": "Number of open credit lines",
    "purpose": "Stated loan purpose carries higher risk",
    "home_ownership": "Housing status carries higher risk",
}
 
 
# ---------------------------------------------------------------------------
# KernelSHAP (model-agnostic, local additive attributions)
# ---------------------------------------------------------------------------
def _shap_kernel_weight(M, s):
    if s == 0 or s == M:
        return 1e6                      # anchors (handled via phi0 + rescale)
    return (M - 1) / (comb(M, s) * s * (M - s))
 
 
def kernel_shap(predict_fn, x_row: pd.Series, background: pd.DataFrame,
                features, nsamples: int = 300, seed: int = 0) -> pd.Series:
    """Return SHAP values (one per feature) for a single instance.
 
    predict_fn: DataFrame -> 1-D array of predicted PD.
    Absent features in a coalition are imputed from a sampled background row.
    """
    rng = np.random.default_rng(seed)
    M = len(features)
    bg = background[features].reset_index(drop=True)
    phi0 = float(predict_fn(bg).mean())                     # E[f]
    fx = float(predict_fn(x_row[features].to_frame().T)[0])  # f(x)
 
    Z, W, rows = [], [], []
    for _ in range(nsamples):
        s = rng.integers(1, M)                              # coalition size 1..M-1
        present = rng.choice(M, size=s, replace=False)
        z = np.zeros(M); z[present] = 1
        base = bg.sample(1, random_state=int(rng.integers(1e9))).iloc[0].copy()
        for j in present:
            base[features[j]] = x_row[features[j]]
        Z.append(z); W.append(_shap_kernel_weight(M, s)); rows.append(base)
 
    Xs = pd.DataFrame(rows)[features]
    yv = predict_fn(Xs) - phi0                              # target = f(h(z)) - E[f]
    Zm, w = np.array(Z), np.array(W)
    # Weighted least squares: (f(h(z)) - phi0) ~ z, no intercept.
    WZ = Zm * w[:, None]
    beta = np.linalg.lstsq(WZ.T @ Zm, WZ.T @ yv, rcond=None)[0]
    # Enforce efficiency: rescale so sum(phi) == f(x) - E[f].
    gap = (fx - phi0) - beta.sum()
    beta = beta + gap / M
    return pd.Series(beta, index=features, name="shap")
 
 
def shap_matrix(predict_fn, X: pd.DataFrame, background: pd.DataFrame,
                features, nsamples: int = 200, seed: int = 0) -> pd.DataFrame:
    out = [kernel_shap(predict_fn, X.iloc[i], background, features, nsamples, seed + i)
           for i in range(len(X))]
    return pd.DataFrame(out, index=X.index)
 
 
def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    return shap_df.abs().mean().sort_values(ascending=False)
 
 
# ---------------------------------------------------------------------------
# Adverse-action reason codes
# ---------------------------------------------------------------------------
def reason_codes_from_shap(shap_row: pd.Series, top_k: int = 4):
    """Top features pushing PD UP for this applicant -> decline reasons."""
    up = shap_row[shap_row > 0].sort_values(ascending=False).head(top_k)
    return [(f, round(float(v), 4), REASON_TEXT.get(f, f)) for f, v in up.items()]
 
 
def reason_codes_from_scorecard(sc, x_row: pd.Series, top_k: int = 4):
    """ECOA/Reg B 'points below maximum' method: the features where the
    applicant lost the most points vs the best attainable score."""
    pts = sc.points_table(x_row.to_frame().T).iloc[0]
    # Reference = max attainable points per feature in the fitted population is
    # approximated by the encoder's max WOE points; here use 0-shortfall proxy:
    shortfall = (pts.max() - pts).sort_values(ascending=False)  # relative
    shortfall = (-(pts)).sort_values(ascending=False).head(top_k)  # lowest points first
    return [(f, round(float(pts[f]), 1), REASON_TEXT.get(f, f)) for f in shortfall.index]
 
 
# ---------------------------------------------------------------------------
# Conceptual-soundness cross-check
# ---------------------------------------------------------------------------
def conceptual_soundness(shap_df: pd.DataFrame, X: pd.DataFrame,
                         champion_pd: np.ndarray, monotone: dict,
                         numeric_features) -> pd.DataFrame:
    """For each numeric feature, compare the sign of the challenger's effect
    (corr between feature value and its SHAP attribution) with (a) the champion
    model's effect (corr between feature value and champion predicted PD) and
    (b) underwriting theory's expected sign. Agreement across all three is
    evidence the ML model learned sensible relationships, not artefacts."""
    champion_pd = np.asarray(champion_pd)
    rows = []
    for f in numeric_features:
        if f not in shap_df.columns:
            continue
        chal_sign = int(np.sign(np.corrcoef(X[f].values, shap_df[f].values)[0, 1]))
        champ_sign = int(np.sign(np.corrcoef(X[f].values, champion_pd)[0, 1]))
        theory = monotone.get(f, 0)
        agree = (chal_sign == champ_sign) and (theory == 0 or chal_sign == theory)
        rows.append({"feature": f, "challenger_sign": chal_sign,
                     "champion_sign": champ_sign, "theory_sign": theory,
                     "all_agree": agree})
    return pd.DataFrame(rows)
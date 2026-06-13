"""
Evaluation metrics across the three families a credit model is judged on:
discrimination, calibration, and stability -- plus a business decision view.
Reporting only AUC is a tell that someone isn't a credit person.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
 
 
# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------
def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1
 
 
def ks_statistic(y, p):
    """Kolmogorov-Smirnov: max separation between cumulative good/bad score
    distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))
 
 
def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"AUC": round(float(auc), 4), "Gini": round(2 * auc - 1, 4),
            "KS": round(ks_statistic(y, p), 4)}
 
 
# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return g.reset_index(drop=True)
 
 
def hosmer_lemeshow(y, p, bins: int = 10) -> dict:
    """HL goodness-of-fit chi-square. Large stat / small p => miscalibration."""
    from scipy import stats
    t = calibration_table(y, p, bins)
    obs_pos, exp_pos = t["actual"] * t["n"], t["predicted"] * t["n"]
    obs_neg, exp_neg = (1 - t["actual"]) * t["n"], (1 - t["predicted"]) * t["n"]
    exp_pos = exp_pos.clip(lower=1e-6); exp_neg = exp_neg.clip(lower=1e-6)
    chi2 = (((obs_pos - exp_pos) ** 2 / exp_pos)
            + ((obs_neg - exp_neg) ** 2 / exp_neg)).sum()
    dof = max(len(t) - 2, 1)
    return {"HL_chi2": round(float(chi2), 2), "dof": int(dof),
            "p_value": round(float(1 - stats.chi2.cdf(chi2, dof)), 4)}
 
 
def calibration(y, p) -> dict:
    out = {"Brier": round(float(brier_score_loss(y, p)), 5)}
    out.update(hosmer_lemeshow(y, p))
    return out
 
 
# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on a score/PD distribution.
    <0.10 stable; 0.10-0.25 some shift; >0.25 significant shift."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
 
 
def psi_report(expected, actual, bins: int = 10) -> dict:
    v = psi(expected, actual, bins)
    band = "stable" if v < 0.10 else ("moderate shift" if v < 0.25 else "significant shift")
    return {"PSI": round(v, 4), "band": band}
 
 
# ---------------------------------------------------------------------------
# Business view
# ---------------------------------------------------------------------------
def approval_badrate_curve(y, p, steps: int = 50) -> pd.DataFrame:
    """Approve the safest applicants; trace approval rate vs realised bad rate
    among the approved population."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p")
    rows = []
    for q in np.linspace(0.05, 1.0, steps):
        k = max(int(len(df) * q), 1)
        appr = df.iloc[:k]
        rows.append({"approval_rate": q, "bad_rate_approved": appr["y"].mean(),
                     "cutoff_pd": appr["p"].max()})
    return pd.DataFrame(rows)
 
 
def expected_loss(p, ead, lgd: float = 0.55) -> float:
    """Portfolio expected loss = sum(PD * LGD * EAD)."""
    return float(np.sum(np.asarray(p) * lgd * np.asarray(ead)))
 
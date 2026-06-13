"""
Champion model: interpretable logistic-regression scorecard.
 
Implements Weight-of-Evidence (WOE) binning, Information Value (IV) ranking,
a logistic regression on WOE-transformed features, and scaling into a
points-based scorecard (base score + PDO). Pure pandas/sklearn so it runs
without optbinning; the API mirrors what optbinning would give you.
 
Also generates adverse-action reason codes from the scorecard via the
"points below maximum" method -- the industry-standard way declines are
explained to applicants under ECOA / Reg B.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
 
 
# ---------------------------------------------------------------------------
# WOE / IV
# ---------------------------------------------------------------------------
def _woe_iv_table(feature: pd.Series, target: pd.Series, bins, is_cat: bool):
    """Build a WOE/IV table for one feature given bin edges or categories."""
    if is_cat:
        grp = feature.astype("object").fillna("NA")
        edges = None
    else:
        grp = pd.cut(feature, bins=bins, include_lowest=True, duplicates="drop")
        edges = bins
    df = pd.DataFrame({"bin": grp, "y": target.values})
    agg = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    agg.columns = ["n", "bad"]
    agg["good"] = agg["n"] - agg["bad"]
    tot_bad, tot_good = agg["bad"].sum(), agg["good"].sum()
    # Laplace smoothing to avoid divide-by-zero in sparse bins.
    dist_bad = (agg["bad"] + 0.5) / (tot_bad + 0.5 * len(agg))
    dist_good = (agg["good"] + 0.5) / (tot_good + 0.5 * len(agg))
    agg["woe"] = np.log(dist_good / dist_bad)
    agg["iv"] = (dist_good - dist_bad) * agg["woe"]
    return agg, edges
 
 
class WOEEncoder:
    """Fit monotone-ish quantile bins per numeric feature, category WOE for
    categoricals, and transform frames into WOE space."""
 
    def __init__(self, numeric, categorical, n_bins: int = 6):
        self.numeric, self.categorical, self.n_bins = numeric, categorical, n_bins
        self.tables_, self.edges_, self.cat_maps_, self.iv_ = {}, {}, {}, {}
 
    def fit(self, X: pd.DataFrame, y: pd.Series):
        for c in self.numeric:
            q = np.unique(np.quantile(X[c].dropna(),
                                      np.linspace(0, 1, self.n_bins + 1)))
            q[0], q[-1] = -np.inf, np.inf
            tbl, _ = _woe_iv_table(X[c], y, q, is_cat=False)
            self.tables_[c], self.edges_[c] = tbl, q
            self.iv_[c] = float(tbl["iv"].sum())
        for c in self.categorical:
            tbl, _ = _woe_iv_table(X[c], y, None, is_cat=True)
            self.tables_[c] = tbl
            self.cat_maps_[c] = tbl["woe"].to_dict()
            self.iv_[c] = float(tbl["iv"].sum())
        return self
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            b = pd.cut(X[c], bins=self.edges_[c], include_lowest=True)
            out[c] = b.map(self.tables_[c]["woe"]).astype(float).fillna(0.0)
        for c in self.categorical:
            default = np.mean(list(self.cat_maps_[c].values()))
            out[c] = (X[c].astype("object").fillna("NA")
                      .map(self.cat_maps_[c]).astype(float).fillna(default))
        return pd.DataFrame(out, index=X.index)
 
    def iv_table(self) -> pd.DataFrame:
        s = (pd.Series(self.iv_, name="IV").sort_values(ascending=False)
             .to_frame())
        s["strength"] = pd.cut(s["IV"], [-1, .02, .1, .3, .5, 1e9],
                               labels=["useless", "weak", "medium", "strong", "suspicious"])
        return s
 
 
# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
class Scorecard:
    """Logistic regression on WOE features, scaled to points.
 
    points = offset - factor * (woe_j * beta_j + intercept/k)
    with factor/offset derived from (base_score, base_odds, PDO).
    """
 
    def __init__(self, encoder: WOEEncoder, base_score=600, base_odds=50, pdo=20):
        self.enc = encoder
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
 
    def fit(self, X: pd.DataFrame, y: pd.Series, min_iv: float = 0.02):
        # Standard scorecard practice: keep predictors with IV >= 0.02.
        all_feats = list(self.enc.numeric) + list(self.enc.categorical)
        self.features_ = [f for f in all_feats if self.enc.iv_.get(f, 0) >= min_iv]
        self.dropped_low_iv_ = [f for f in all_feats if f not in self.features_]
        W = self.enc.transform(X)[self.features_]
        self.lr_ = LogisticRegression(max_iter=1000, C=1e6)  # near-unregularised
        self.lr_.fit(W, y)
        self.coef_ = dict(zip(self.features_, self.lr_.coef_[0]))
        self.intercept_ = float(self.lr_.intercept_[0])
        self._wald(W, y)
        return self
 
    def _wald(self, W, y):
        """Wald std errors / p-values via (X'VX)^-1 (replaces statsmodels)."""
        from scipy import stats
        Xd = np.column_stack([np.ones(len(W)), W.values])
        p = np.clip(self.lr_.predict_proba(W)[:, 1], 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        try:
            # X' V X without materialising the n x n diagonal matrix.
            XtVX = Xd.T @ (Xd * w[:, None])
            XtVX += 1e-8 * np.eye(XtVX.shape[0])     # ridge for stability
            cov = np.linalg.inv(XtVX)
            se = np.sqrt(np.diag(cov))
            beta = np.concatenate([[self.intercept_], self.lr_.coef_[0]])
            zstat = beta / se
            pval = 2 * (1 - stats.norm.cdf(np.abs(zstat)))
            names = ["intercept"] + self.features_
            self.wald_ = pd.DataFrame({"coef": beta, "std_err": se,
                                       "z": zstat, "p_value": pval}, index=names)
        except np.linalg.LinAlgError:
            self.wald_ = None
 
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        W = self.enc.transform(X)[self.features_]
        return self.lr_.predict_proba(W)[:, 1]
 
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Total scorecard points (higher = safer)."""
        W = self.enc.transform(X)[self.features_]
        k = len(self.features_)
        pts = np.zeros(len(X))
        for j, c in enumerate(self.features_):
            pts += -self.factor * (W[c].values * self.coef_[c]
                                   + self.intercept_ / k)
        return self.offset + pts
 
    def points_table(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature points for each row (for reason codes & transparency)."""
        W = self.enc.transform(X)[self.features_]
        k = len(self.features_)
        cols = {c: -self.factor * (W[c].values * self.coef_[c] + self.intercept_ / k)
                for c in self.features_}
        return pd.DataFrame(cols, index=X.index)
 
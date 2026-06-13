"""
Challenger model: gradient-boosted trees with MONOTONIC CONSTRAINTS + calibration.
 
Uses sklearn HistGradientBoostingClassifier, which natively supports monotonic
constraints (monotonic_cst) -- the same technique XGBoost exposes via
`monotone_constraints`. Swap in XGBoost by replacing the estimator; the
constraint vector and calibration wrapper are identical in spirit.
 
Why constraints: regulators expect economically sensible behaviour, e.g. higher
DTI must never *lower* predicted risk. Why calibration: PD *levels* (not just
ranking) drive pricing and expected-loss provisioning, so we recalibrate rather
than oversample/SMOTE (which distorts PD levels).
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
 
 
def build_challenger(numeric, categorical, monotone: dict,
                     calibrate: bool = True, calib_method: str = "isotonic",
                     random_state: int = 7):
    """Return an (uncalibrated, calibrated_or_not) sklearn pipeline.
 
    monotone: {feature: +1 / -1 / 0} in *risk* direction. We model P(default),
    so the sign passes straight through to monotonic_cst.
    """
    feats = list(numeric) + list(categorical)
    # Ordinal-encode categoricals; HGB handles them as numeric codes with
    # unconstrained (0) monotonicity.
    pre = ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), categorical)],
        remainder="drop")
    mono_vec = [int(monotone.get(f, 0)) for f in numeric] + [0] * len(categorical)
 
    hgb = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=200,
        monotonic_cst=mono_vec, class_weight="balanced",  # ~scale_pos_weight
        random_state=random_state)
 
    base = Pipeline([("pre", pre), ("hgb", hgb)])
    if not calibrate:
        return base, feats, mono_vec
    calibrated = CalibratedClassifierCV(base, method=calib_method, cv=3)
    return calibrated, feats, mono_vec
 
 
def check_monotonicity(model, X: pd.DataFrame, feature: str, direction: int,
                       grid: int = 25) -> bool:
    """Empirically verify the fitted model is monotone in `feature`:
    sweep the feature across its range on a fixed background, check the
    predicted-PD curve never moves the wrong way."""
    if direction == 0:
        return True
    bg = X.sample(min(400, len(X)), random_state=0).copy()
    xs = np.linspace(X[feature].quantile(.02), X[feature].quantile(.98), grid)
    means = []
    for v in xs:
        bg[feature] = v
        means.append(model.predict_proba(bg)[:, 1].mean())
    diffs = np.diff(means)
    # direction +1 => PD should be non-decreasing; -1 => non-increasing.
    tol = 1e-6
    return bool((diffs >= -tol).all()) if direction > 0 else bool((diffs <= tol).all())
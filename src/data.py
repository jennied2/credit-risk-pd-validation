"""
Data layer for the Explainable PD model + fair-lending audit.
 
Design goal: the pipeline runs end-to-end with NO network and NO real data,
but transparently switches to real files if you provide them.
 
Lending Club  -> probability-of-default (PD) modelling. Has realised outcomes,
                 no protected attributes.
HMDA          -> fair-lending audit. Has applicant race/ethnicity/sex and the
                 approve/deny decision, but no default outcome.
 
To use REAL data, drop the files here and re-run; loaders prefer them:
    data/lending_club.csv   (e.g. Kaggle "Lending Club Loan Data")
    data/hmda_2024.csv      (FFIEC HMDA Data Browser slice: one state/year,
                             first-lien owner-occupied 1-4 family conventional
                             purchase loans)
"""
from __future__ import annotations
 
import os
import numpy as np
import pandas as pd
 
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
 
# ---------------------------------------------------------------------------
# Column governance: what the model may see, and what must never touch it.
# ---------------------------------------------------------------------------
 
# Pure-underwriting features (the model we actually build).
LC_UNDERWRITING_FEATURES = [
    "fico", "dti", "annual_inc", "loan_amnt", "term_months", "emp_length",
    "revol_util", "inq_last_6mths", "open_acc", "purpose", "home_ownership",
]
LC_NUMERIC = ["fico", "dti", "annual_inc", "loan_amnt", "term_months",
              "emp_length", "revol_util", "inq_last_6mths", "open_acc"]
LC_CATEGORICAL = ["purpose", "home_ownership"]
 
# Lending Club's OWN risk assessment. Including these means modelling LC's
# pricing rather than raw creditworthiness -> excluded from the primary model.
LC_LENDER_ASSESSMENT = ["grade", "sub_grade", "int_rate"]
 
# Post-origination / outcome-encoding fields -> LABEL LEAKAGE. Must be dropped.
LC_LEAKAGE = [
    "recoveries", "collection_recovery_fee", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "last_pymnt_amnt", "out_prncp", "funded_amnt",
]
 
# Monotone direction expected by underwriting theory (+1 risk rises with feature,
# -1 risk falls). Used for monotonic constraints and the sign sanity check.
LC_MONOTONE = {
    "fico": -1, "dti": +1, "annual_inc": -1, "loan_amnt": +1, "term_months": +1,
    "emp_length": -1, "revol_util": +1, "inq_last_6mths": +1, "open_acc": 0,
}
 
 
# ---------------------------------------------------------------------------
# Synthetic Lending Club generator
# ---------------------------------------------------------------------------
def _make_synthetic_lending_club(n: int = 60_000, seed: int = 7) -> pd.DataFrame:
    """Realistic LC-style loan tape with injected leakage, vintage drift and
    monotone+nonlinear risk so the ML challenger can genuinely beat the GLM."""
    rng = np.random.default_rng(seed)
 
    fico = np.clip(rng.normal(695, 45, n), 610, 845).round(0)
    annual_inc = np.clip(rng.lognormal(11.0, 0.55, n), 12_000, 400_000).round(0)
    dti = np.clip(rng.normal(18, 8, n), 0, 45).round(2)
    loan_amnt = np.clip(rng.normal(15_000, 8_500, n), 1_000, 40_000).round(0)
    term_months = rng.choice([36, 60], n, p=[0.72, 0.28])
    emp_length = rng.integers(0, 11, n)
    revol_util = np.clip(rng.normal(52, 24, n), 0, 135).round(1)
    inq_last_6mths = rng.poisson(0.7, n)
    open_acc = np.clip(rng.normal(11, 5, n), 1, 40).round(0)
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "other"],
        n, p=[0.46, 0.22, 0.10, 0.08, 0.04, 0.10])
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.40, 0.48, 0.12])
 
    # Origination dates 2014-01 .. 2018-12 (vintages).
    issue_month = rng.integers(0, 60, n)               # months since 2014-01
    issue_year = 2014 + issue_month // 12
 
    # Mild credit-box loosening over time (as observed 2014->2018): DTI creeps
    # up, FICO drifts down. Drives a realistic population shift (PSI) for the
    # ongoing-monitoring section.
    yr = issue_year - 2014
    dti = np.clip(dti + 0.9 * yr, 0, 45).round(2)
    fico = np.clip(fico - 4.0 * yr, 610, 845).round(0)
 
    purpose_risk = pd.Series(purpose).map(
        {"debt_consolidation": 0.10, "credit_card": 0.05, "home_improvement": -0.10,
         "major_purchase": 0.00, "small_business": 0.55, "other": 0.15}).to_numpy()
    home_risk = pd.Series(home_ownership).map(
        {"RENT": 0.18, "MORTGAGE": -0.12, "OWN": -0.02}).to_numpy()
    # Macro/vintage effect: 2016 book underperforms (cycle), drifts the OOT score.
    vintage_risk = np.where(issue_year == 2016, 0.35, 0.0) + 0.04 * (issue_year - 2014)
 
    z = (-3.75
         + 0.017 * (680 - fico)
         + 0.050 * dti
         + 0.024 * revol_util
         - 6.0e-6 * annual_inc
         + 0.32 * inq_last_6mths
         + 0.50 * (term_months == 60)
         - 0.03 * emp_length
         + 0.020 * dti * (term_months == 60) / 10        # nonlinear interaction
         + 0.60 * (revol_util > 90)                       # nonlinear kink
         + purpose_risk + home_risk + vintage_risk
         + rng.normal(0, 0.35, n))
    pd_true = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, pd_true)
 
    # LC's own assessment, correlated with true risk (so we can show the tradeoff).
    grade_idx = np.clip(np.digitize(z, np.quantile(z, [.14, .30, .50, .70, .86, .95])), 0, 6)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    int_rate = (6.0 + 3.1 * grade_idx + rng.normal(0, 0.8, n)).round(2)
 
    # Loan status, including records that MUST be filtered out.
    # Observation date assumed ~2023-01 (month 108 from 2014-01), so most loans
    # have matured; recent 60-month loans have NOT and must be dropped.
    obs_month = 108
    months_on_book = obs_month - issue_month
    matured = months_on_book >= term_months              # reached full term
    status = np.where(default == 1,
                      rng.choice(["Charged Off", "Default"], n, p=[0.85, 0.15]),
                      "Fully Paid")
    # Immature loans: outcome not yet known -> Current / In Grace Period.
    immature = ~matured
    status = np.where(immature & (rng.random(n) < 0.85), "Current", status)
    status = np.where(immature & (status != "Current"), "In Grace Period", status)
 
    # Post-origination / outcome fields -> LEAKAGE (injected on purpose).
    total_pymnt = np.where(default == 1,
                           loan_amnt * rng.uniform(0.10, 0.70, n),
                           loan_amnt * rng.uniform(1.02, 1.35, n)).round(2)
    recoveries = np.where(default == 1, loan_amnt * rng.uniform(0.0, 0.25, n), 0.0).round(2)
    out_prncp = np.where(status == "Current", loan_amnt * rng.uniform(0.3, 0.9, n), 0.0).round(2)
 
    df = pd.DataFrame({
        "id": np.arange(n),
        "issue_d": pd.to_datetime("2014-01-01") + pd.to_timedelta(issue_month * 30, "D"),
        "issue_year": issue_year,
        "fico": fico, "annual_inc": annual_inc, "dti": dti, "loan_amnt": loan_amnt,
        "term_months": term_months, "emp_length": emp_length, "revol_util": revol_util,
        "inq_last_6mths": inq_last_6mths, "open_acc": open_acc,
        "purpose": purpose, "home_ownership": home_ownership,
        "grade": grade, "sub_grade": grade, "int_rate": int_rate,
        "loan_status": status,
        # leakage:
        "total_pymnt": total_pymnt, "total_rec_prncp": (total_pymnt * 0.8).round(2),
        "total_rec_int": (total_pymnt * 0.2).round(2),
        "recoveries": recoveries, "collection_recovery_fee": (recoveries * 0.1).round(2),
        "last_pymnt_amnt": (total_pymnt / np.maximum(term_months, 1)).round(2),
        "out_prncp": out_prncp, "funded_amnt": loan_amnt,
        "_pd_true": pd_true,    # oracle, for diagnostics only; never a feature
    })
    return df
 
 
def load_lending_club(prefer_real: bool = True) -> tuple[pd.DataFrame, bool]:
    """Return (raw_df, is_real). Uses data/lending_club.csv if present."""
    path = os.path.join(DATA_DIR, "lending_club.csv")
    if prefer_real and os.path.exists(path):
        return pd.read_csv(path, low_memory=False), True
    return _make_synthetic_lending_club(), False
 
 
def define_target_and_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the credit target definition, maturity filter and leakage audit.
 
    Returns the modelling frame and an audit dict for the validation report.
    """
    audit: dict = {}
    n0 = len(df)
 
    # 1) Target definition.
    bad = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
    good = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
    drop_status = ~df["loan_status"].isin(bad | good)      # Current / In Grace / etc.
    audit["rows_total"] = n0
    audit["rows_dropped_unresolved"] = int(drop_status.sum())
    audit["unresolved_status_values"] = (
        df.loc[drop_status, "loan_status"].value_counts().to_dict())
 
    work = df.loc[~drop_status].copy()
    work["default"] = work["loan_status"].isin(bad).astype(int)
    audit["rows_kept"] = len(work)
    audit["default_rate"] = round(float(work["default"].mean()), 4)
 
    # 2) Leakage audit: identify and drop post-origination / outcome fields.
    present_leakage = [c for c in LC_LEAKAGE if c in work.columns]
    audit["leakage_columns_dropped"] = present_leakage
    audit["lender_assessment_excluded"] = [c for c in LC_LENDER_ASSESSMENT if c in work.columns]
 
    # Keep underwriting features + target + date + (excluded fields stay only
    # for the leakage demonstration cell, dropped from the modelling matrix).
    keep = (["id", "issue_d", "issue_year", "default"]
            + [c for c in LC_UNDERWRITING_FEATURES if c in work.columns])
    leak_demo = [c for c in present_leakage if c in work.columns]
    model_df = work[keep].copy()
    leak_df = work[keep + leak_demo].copy()             # for the leakage demo only
 
    return model_df, {"audit": audit, "leak_demo_df": leak_df, "leak_cols": leak_demo}
 
 
def oot_split(df: pd.DataFrame, train_years=(2014, 2015, 2016),
              valid_years=(2017,), oot_years=(2018,)):
    """Out-of-time split: train on older vintages, validate, hold out a LATER
    vintage as the OOT test. Mirrors how PD models are actually validated."""
    tr = df[df["issue_year"].isin(train_years)].copy()
    va = df[df["issue_year"].isin(valid_years)].copy()
    oot = df[df["issue_year"].isin(oot_years)].copy()
    return tr, va, oot
 
 
# ---------------------------------------------------------------------------
# Synthetic HMDA generator (fair-lending audit; no default outcome)
# ---------------------------------------------------------------------------
HMDA_FEATURES = ["loan_amount", "property_value", "income",
                 "debt_to_income_ratio", "loan_to_value_ratio", "loan_term"]
HMDA_PROTECTED = ["derived_race", "derived_sex", "derived_ethnicity"]
 
 
def _make_synthetic_hmda(n: int = 40_000, seed: int = 11) -> pd.DataFrame:
    """HMDA-style application records for one state/year slice: first-lien,
    owner-occupied, 1-4 family, conventional purchase. Embeds a residual group
    disparity on top of legitimate risk factors so the audit detects impact."""
    rng = np.random.default_rng(seed)
 
    race = rng.choice(
        ["White", "Black or African American", "Asian",
         "American Indian or Alaska Native", "Joint"],
        n, p=[0.62, 0.16, 0.13, 0.03, 0.06])
    ethnicity = rng.choice(["Not Hispanic or Latino", "Hispanic or Latino"],
                           n, p=[0.80, 0.20])
    sex = rng.choice(["Male", "Female", "Joint"], n, p=[0.45, 0.33, 0.22])
 
    income = np.clip(rng.lognormal(11.2, 0.5, n), 18_000, 500_000)
    property_value = np.clip(rng.lognormal(12.6, 0.45, n), 60_000, 2_000_000)
    dti = np.clip(rng.normal(36, 9, n), 8, 65)
    ltv_base = rng.uniform(0.55, 0.97, n)
 
    # Structural correlation between protected status and observable factors
    # (reflects real wealth/income gaps): disadvantaged groups have, on average,
    # lower income and higher DTI / LTV. This makes the legitimate features
    # PROXY race -- so a model that never sees race still reproduces disparity,
    # which is the crux of disparate-impact analysis.
    inc_mult = (np.where(race == "Black or African American", 0.80, 1.0)
                * np.where(race == "American Indian or Alaska Native", 0.82, 1.0)
                * np.where(ethnicity == "Hispanic or Latino", 0.86, 1.0)
                * np.where(sex == "Female", 0.95, 1.0))
    dti_add = (np.where(race == "Black or African American", 4.0, 0.0)
               + np.where(race == "American Indian or Alaska Native", 4.0, 0.0)
               + np.where(ethnicity == "Hispanic or Latino", 2.5, 0.0))
    ltv_add = (np.where(race == "Black or African American", 0.05, 0.0)
               + np.where(race == "American Indian or Alaska Native", 0.05, 0.0)
               + np.where(ethnicity == "Hispanic or Latino", 0.035, 0.0))
    income = np.clip(income * inc_mult, 18_000, 500_000).round(0)
    dti = np.clip(dti + dti_add, 8, 65).round(1)
 
    property_value = np.clip(property_value, 60_000, 2_000_000).round(-3)
    loan_amount = np.clip(property_value * np.clip(ltv_base + ltv_add, 0.30, 0.99),
                          30_000, None).round(-3)
    ltv = np.clip(loan_amount / property_value * 100, 30, 100).round(1)
    loan_term = rng.choice([180, 360], n, p=[0.18, 0.82])
 
    # Residual direct disparity NOT explained by legitimate factors (smaller now
    # that much of the gap flows through the proxy features above).
    grp = (np.where(race == "Black or African American", 0.25, 0.0)
           + np.where(race == "American Indian or Alaska Native", 0.20, 0.0)
           + np.where(ethnicity == "Hispanic or Latino", 0.15, 0.0)
           + np.where(sex == "Female", 0.05, 0.0))
 
    z = (-3.1
         + 0.045 * dti
         + 0.030 * ltv
         - 9.0e-6 * income
         + 6.0e-7 * loan_amount
         + grp
         + rng.normal(0, 0.4, n))
    deny = rng.binomial(1, 1 / (1 + np.exp(-z)))
    action_taken = np.where(deny == 1, 3, 1)            # 3 denied, 1 originated
 
    return pd.DataFrame({
        "loan_amount": loan_amount, "property_value": property_value, "income": income,
        "debt_to_income_ratio": dti, "loan_to_value_ratio": ltv, "loan_term": loan_term,
        "derived_race": race, "derived_ethnicity": ethnicity, "derived_sex": sex,
        "action_taken": action_taken,
        "lien_status": 1, "occupancy_type": 1, "loan_purpose": 1,   # filtered slice
    })
 
 
def load_hmda(prefer_real: bool = True) -> tuple[pd.DataFrame, bool]:
    """Return (df, is_real). Uses data/hmda_2024.csv if present.
    Expected slice: first-lien owner-occupied 1-4 family conventional purchase."""
    path = os.path.join(DATA_DIR, "hmda_2024.csv")
    if prefer_real and os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        # Keep only resolved approve/deny decisions for the audit.
        df = df[df["action_taken"].isin([1, 3])].copy()
        return df, True
    return _make_synthetic_hmda(), False
 
 
def hmda_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary audit target: denied = 1 (adverse), originated = 0."""
    df = df.copy()
    df["denied"] = (df["action_taken"] == 3).astype(int)
    for c in HMDA_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=HMDA_FEATURES)
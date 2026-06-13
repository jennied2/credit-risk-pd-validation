"""
One-time prep: convert the raw Kaggle Lending Club file
(accepted_2007_to_2018Q4.csv or .csv.gz) into data/lending_club.csv with the
clean schema the pipeline expects.

Usage:
    python scripts/prepare_lending_club.py /path/to/accepted_2007_to_2018Q4.csv.gz

The pipeline then auto-detects data/lending_club.csv and recomputes on real data.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

# ---- knobs ---------------------------------------------------------------
KEEP_YEARS = set(range(2014, 2019))   # 2014..2018 (works with the default split)
SAMPLE_N   = 250_000                  # set to None to keep ALL rows (slower)
SEED       = 7
# --------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "data", "lending_club.csv")

# raw columns we actually need (everything else is ignored to save memory)
NEEDED = {
    "id", "issue_d", "loan_amnt", "funded_amnt", "term", "int_rate", "grade",
    "sub_grade", "emp_length", "home_ownership", "annual_inc", "purpose", "dti",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "revol_util",
    "loan_status", "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries",
    "collection_recovery_fee", "last_pymnt_amnt", "out_prncp",
}

def pct_to_float(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(),
                         errors="coerce")

def emp_to_years(s):
    s = s.astype(str)
    out = s.str.extract(r"(\d+)")[0].astype(float)        # "10+ years"->10, "3 years"->3
    out = out.where(~s.str.contains("<", na=False), 0.0)  # "< 1 year"->0
    return out

def term_to_months(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")

def main(src):
    print(f"reading {src} (this can take a minute) ...")
    raw = pd.read_csv(src, compression="infer", low_memory=False,
                      usecols=lambda c: c in NEEDED, on_bad_lines="skip")
    print(f"  raw rows: {len(raw):,}  raw cols: {raw.shape[1]}")

    iss = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    miss = iss.isna()
    if miss.any():   # fall back for any odd date formats
        iss = iss.fillna(pd.to_datetime(raw["issue_d"], errors="coerce"))

    df = pd.DataFrame({
        "id": raw.get("id", pd.Series(np.arange(len(raw)))),
        "issue_d": iss,
        "issue_year": iss.dt.year,
        "fico": (pd.to_numeric(raw["fico_range_low"], errors="coerce")
                 + pd.to_numeric(raw["fico_range_high"], errors="coerce")) / 2.0,
        "dti": pd.to_numeric(raw["dti"], errors="coerce"),
        "annual_inc": pd.to_numeric(raw["annual_inc"], errors="coerce"),
        "loan_amnt": pd.to_numeric(raw["loan_amnt"], errors="coerce"),
        "term_months": term_to_months(raw["term"]),
        "emp_length": emp_to_years(raw["emp_length"]),
        "revol_util": pct_to_float(raw["revol_util"]),
        "inq_last_6mths": pd.to_numeric(raw["inq_last_6mths"], errors="coerce"),
        "open_acc": pd.to_numeric(raw["open_acc"], errors="coerce"),
        "purpose": raw["purpose"].astype(str),
        "home_ownership": raw["home_ownership"].astype(str),
        "grade": raw["grade"].astype(str),
        "sub_grade": raw["sub_grade"].astype(str),
        "int_rate": pct_to_float(raw["int_rate"]),
        "loan_status": raw["loan_status"].astype(str),
        # leakage fields (kept so the leakage-audit demo has something to find)
        "total_pymnt": pd.to_numeric(raw["total_pymnt"], errors="coerce"),
        "total_rec_prncp": pd.to_numeric(raw["total_rec_prncp"], errors="coerce"),
        "total_rec_int": pd.to_numeric(raw["total_rec_int"], errors="coerce"),
        "recoveries": pd.to_numeric(raw["recoveries"], errors="coerce"),
        "collection_recovery_fee": pd.to_numeric(raw["collection_recovery_fee"], errors="coerce"),
        "last_pymnt_amnt": pd.to_numeric(raw["last_pymnt_amnt"], errors="coerce"),
        "out_prncp": pd.to_numeric(raw["out_prncp"], errors="coerce"),
    })

    # restrict to the vintages the split uses
    df = df[df["issue_year"].isin(KEEP_YEARS)].copy()

    # median-fill the handful of NaNs in numeric underwriting fields so the
    # WOE binning doesn't drop rows (HistGBM tolerates NaN, the scorecard less so)
    for col in ["fico", "dti", "annual_inc", "loan_amnt", "term_months",
                "emp_length", "revol_util", "inq_last_6mths", "open_acc"]:
        df[col] = df[col].fillna(df[col].median())
    df["term_months"] = df["term_months"].round().astype(int)

    # drop rows with no usable status or date
    df = df.dropna(subset=["issue_year", "loan_status"])

    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(f"  rows: {len(df):,}")
    print(f"  by issue_year:\n{df['issue_year'].value_counts().sort_index().to_string()}")
    resolved = df["loan_status"].isin(
        {"Fully Paid", "Charged Off", "Default",
         "Does not meet the credit policy. Status:Fully Paid",
         "Does not meet the credit policy. Status:Charged Off"}).sum()
    print(f"  resolved (model-eligible) rows: {resolved:,}")
    print("\nDone. Now re-run:  python src/run_pipeline.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/prepare_lending_club.py <path-to-accepted_*.csv[.gz]>")
    main(sys.argv[1])

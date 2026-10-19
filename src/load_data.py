"""
Loads NHANES August 2021 - August 2023 (the most recent cycle) as a model-ready DataFrame.

    from load_data import load_nhanes, TARGET
    df = load_nhanes()
    X, y = df.drop(columns=TARGET), df[TARGET]

Missing values stay as NaN: fill them in after the train/test split.
"""
import os
import urllib.request

import numpy as np
import pyreadstat

URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nhanes_2021_2023")
FILES = ["DEMO_L", "GHB_L", "DIQ_L", "BMX_L", "BPXO_L", "DR1TOT_L",
         "PAQ_L", "SMQ_L", "SLQ_L", "ALQ_L", "MCQ_L"]

TARGET = "risk"   # 0 = low, 1 = prediabetes, 2 = diabetes
NUMERIC_COLS = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "waist_height_ratio", "sbp_mean", "dbp_mean",
                "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT",
                "equivalent_min", "PAD680", "SLD012", "smoking_status", "alcohol_per_week"]
BINARY_COLS = ["RIAGENDR", "ALQ111", "cardio_history"]


def read_file(name):
    """
    Downloads an NHANES file the first time (cached in data/), then reads it.
    Returns it as a DataFrame.
    """
    path = os.path.join(DATA_DIR, name + ".xpt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(URL + name + ".xpt", path)
    df, _ = pyreadstat.read_xport(path, encoding="windows-1252")
    return df


def minutes_per_week(times, unit, minutes):
    """
    Leisure activity: times per day/week/month/year x minutes each time -> minutes per week.
    Returns 0 for people who never do it.
    """
    per_week = times.replace([7777, 9999], np.nan) * unit.map({"D": 7, "W": 1, "M": 12 / 52, "Y": 1 / 52})
    return (per_week * minutes.replace([7777, 9999], np.nan)).mask(times == 0, 0)

def clean_data(df):
    """
    Cleans the NHANES data by removing rows with missing values in critical columns.
    Returns the cleaned DataFrame.
    """
    # Remove rows with missing values in critical columns
    critical_columns = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "waist_height_ratio", "sbp_mean", "dbp_mean",
                        "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT",
                        "equivalent_min", "PAD680", "SLD012", "smoking_status", "alcohol_per_week",
                        "RIAGENDR", "ALQ111", "cardio_history"]
    df_cleaned = df.dropna(subset=critical_columns)
    return df_cleaned

def load_nhanes():
    """
    Returns one row per adult (index = SEQN): the features plus the risk label.
    Codes like 7/9, 777/999, 7777/9999 mean refused / don't know and become NaN.
    """
    # 1. Join every file on the participant ID
    df = read_file(FILES[0])
    for name in FILES[1:]:
        df = df.merge(read_file(name), on="SEQN", how="left")

    # 2. Keep adults (20+) who did the exam, have an HbA1c result and are not pregnant
    df = df[(df.RIDAGEYR >= 20) & (df.RIDSTATR == 2) & df.LBXGH.notna() & (df.RIDEXPRG != 1)]
    df = df.set_index(df.SEQN.astype(int))

    # 3. Features
    X = df[["RIDAGEYR", "INDFMPIR", "BMXBMI", "SLD012",
            "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT"]].copy()
    X["RIAGENDR"] = df.RIAGENDR.map({1: 0, 2: 1})                      # 0 = male, 1 = female
    X["waist_height_ratio"] = df.BMXWAIST / df.BMXHT
    X["sbp_mean"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)  # mean of 3 readings
    X["dbp_mean"] = df[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    X["PAD680"] = df.PAD680.replace([7777, 9999], np.nan)               # minutes sitting per day

    # moderate + 2 x vigorous minutes per week (WHO convention)
    X["equivalent_min"] = (minutes_per_week(df.PAD790Q, df.PAD790U, df.PAD800)
                           + 2 * minutes_per_week(df.PAD810Q, df.PAD810U, df.PAD820))

    # 0 = never smoked, 1 = former smoker, 2 = current smoker
    X["smoking_status"] = np.select([df.SMQ020 == 2, df.SMQ040.isin([1, 2]), df.SMQ040 == 3],
                                    [0, 2, 1], default=np.nan)

    # drinks per week = drinking days per week x drinks per day (0 if never drank or not in the past year)
    X["ALQ111"] = df.ALQ111.map({1: 1, 2: 0})                          # 1 = ever had a drink
    days_per_week = df.ALQ121.map({0: 0, 1: 7, 2: 6, 3: 3.5, 4: 2, 5: 1, 6: 2.5 * 12 / 52,
                                   7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52})
    drinks = days_per_week * df.ALQ130.replace([777, 999], np.nan)
    X["alcohol_per_week"] = drinks.mask(days_per_week == 0, 0).mask(df.ALQ111 == 2, 0)

    # 1 = ever told heart failure, coronary disease, angina, heart attack or stroke
    heart = df[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    X["cardio_history"] = np.select([(heart == 1).any(axis=1), (heart == 2).all(axis=1)],
                                    [1, 0], default=np.nan)

    # 4. Label: HbA1c >= 6.5 or diagnosed -> 2, HbA1c 5.7-6.4 or "borderline" -> 1, else 0
    X = X[NUMERIC_COLS + BINARY_COLS].astype(float)
    X[TARGET] = np.select([(df.LBXGH >= 6.5) | (df.DIQ010 == 1),
                           (df.LBXGH >= 5.7) | (df.DIQ010 == 3)], [2, 1], default=0)
    return X


if __name__ == "__main__":
    df = load_nhanes()
    print(len(df[df["smoking_status"].isna()]))
    print(len(df))
    df = clean_data(df)
    print(len(df[df["smoking_status"].isna()]))
    print(len(df))
    print(df.shape)
    print(df[TARGET].value_counts().sort_index())

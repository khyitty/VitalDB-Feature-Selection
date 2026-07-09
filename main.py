import os

import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm


ALL_CASEIDS = list(range(1, 6389))
DEFAULT_N_CASES = 100

TRACKS = [
    "BIS/BIS",
    "BIS/SQI",
    "Solar8000/HR",
    "Solar8000/PLETH_HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/FEM_MBP",
    "Solar8000/FEM_SBP",
    "Solar8000/FEM_DBP",
    "Primus/ETCO2",
    "Solar8000/ETCO2",
    "Orchestra/PPF20_CE",
    "Orchestra/PPF20_CP",
    "Orchestra/PPF20_RATE",
    "Orchestra/PPF20_VOL",
    "Orchestra/RFTN20_CE",
    "Orchestra/RFTN20_CP",
    "Orchestra/RFTN20_RATE",
    "Orchestra/RFTN20_VOL",
]

REQUIRED_TRACKS = [
    "BIS/BIS",
    "BIS/SQI",
    "Solar8000/HR",
    "Orchestra/PPF20_CE",
    "Orchestra/PPF20_RATE",
]

TS_COLS = [
    "SQI",
    "HR",
    "MBP",
    "SBP",
    "DBP",
    "SPO2",
    "ETCO2",
    "PPF_CE",
    "PPF_CP",
    "PPF_RATE",
    "PPF_VOL",
    "RFTN_CE",
    "RFTN_CP",
    "RFTN_RATE",
    "RFTN_VOL",
]

CLINICAL_COLS = [
    "caseid",
    "age",
    "sex",
    "height",
    "weight",
    "bmi",
    "asa",
    "emop",
    "optype",
    "ane_type",
    "anestart",
    "aneend",
    "opstart",
    "opend",
    "intraop_ppf",
    "intraop_ftn",
    "intraop_mdz",
    "intraop_rocu",
    "intraop_eph",
    "intraop_phe",
]


def display(obj):
    print(obj)


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def load_clinical():
    # vitaldb 1.7.2 returns an empty dataframe when caseids is omitted.
    clinical = vitaldb.load_clinical_data(caseids=ALL_CASEIDS)
    clinical.to_csv("data/raw/clinical.csv", index=False)
    return clinical


def load_cases(caseids, tracks):
    all_dfs = []

    for cid in tqdm(caseids, desc="Loading VitalDB cases"):
        try:
            arr = vitaldb.load_case(int(cid), ",".join(tracks), interval=1.0)
            temp = pd.DataFrame(arr, columns=tracks)
            temp["caseid"] = int(cid)
            temp["time_sec"] = np.arange(len(temp))
            all_dfs.append(temp)
        except Exception as exc:
            print(f"case {cid} failed: {exc}")

    if not all_dfs:
        raise RuntimeError("No cases were loaded.")

    return pd.concat(all_dfs, ignore_index=True)


def first_available(raw, columns):
    return raw.reindex(columns=columns).bfill(axis=1).iloc[:, 0]


def build_fallback_dataset(raw):
    df = pd.DataFrame()
    df["caseid"] = raw["caseid"]
    df["time_sec"] = raw["time_sec"]
    df["BIS"] = raw["BIS/BIS"]
    df["SQI"] = raw["BIS/SQI"]
    df["HR"] = first_available(raw, ["Solar8000/HR", "Solar8000/PLETH_HR"])
    df["MBP"] = first_available(raw, ["Solar8000/ART_MBP", "Solar8000/FEM_MBP", "Solar8000/NIBP_MBP"])
    df["SBP"] = first_available(raw, ["Solar8000/ART_SBP", "Solar8000/FEM_SBP", "Solar8000/NIBP_SBP"])
    df["DBP"] = first_available(raw, ["Solar8000/ART_DBP", "Solar8000/FEM_DBP", "Solar8000/NIBP_DBP"])
    df["SPO2"] = raw["Solar8000/PLETH_SPO2"]
    df["ETCO2"] = first_available(raw, ["Primus/ETCO2", "Solar8000/ETCO2"])
    df["PPF_CE"] = raw["Orchestra/PPF20_CE"]
    df["PPF_CP"] = raw["Orchestra/PPF20_CP"]
    df["PPF_RATE"] = raw["Orchestra/PPF20_RATE"]
    df["PPF_VOL"] = raw["Orchestra/PPF20_VOL"]
    df["RFTN_CE"] = raw["Orchestra/RFTN20_CE"]
    df["RFTN_CP"] = raw["Orchestra/RFTN20_CP"]
    df["RFTN_RATE"] = raw["Orchestra/RFTN20_RATE"]
    df["RFTN_VOL"] = raw["Orchestra/RFTN20_VOL"]
    return df


def filter_signal_quality(df):
    before = len(df)
    df = df[df["BIS"].notna()].copy()
    df = df[(df["BIS"] >= 10) & (df["BIS"] <= 100)].copy()
    df = df[(df["SQI"].isna()) | (df["SQI"] >= 50)].copy()
    after = len(df)
    print(f"\nBIS/SQI filtering: {before:,} -> {after:,} rows kept ({after / before:.3f})")
    return df


def crop_to_propofol_period(group):
    group = group.sort_values("time_sec").copy()
    active = (
        (group["PPF_CE"].fillna(0) > 0.05)
        | (group["PPF_CP"].fillna(0) > 0.05)
        | (group["PPF_RATE"].fillna(0) > 0)
    )

    if active.sum() == 0:
        return group.iloc[0:0]

    start = group.loc[active, "time_sec"].min()
    end = group.loc[active, "time_sec"].max()
    return group[(group["time_sec"] >= start) & (group["time_sec"] <= end)].copy()


def crop_and_filter_cases(df):
    before = len(df)
    cropped = [crop_to_propofol_period(group) for _, group in df.groupby("caseid", sort=False)]
    df = pd.concat(cropped, ignore_index=True) if cropped else df.iloc[0:0].copy()
    after = len(df)
    print(f"Propofol-period crop: {before:,} -> {after:,} rows kept ({after / before:.3f})")

    case_lengths = df.groupby("caseid").size()
    keep_cases = case_lengths[case_lengths >= 300].index
    df = df[df["caseid"].isin(keep_cases)].copy()
    print("cases after minimum length filter:", df["caseid"].nunique())
    print("rows after minimum length filter:", len(df))
    return df


def encode_sex(value):
    sex = str(value).strip().lower()
    if sex in ["m", "male", "1"]:
        return 1
    if sex in ["f", "female", "0"]:
        return 0
    return np.nan


def add_clinical_and_labels(df, clinical):
    clinical_use = clinical[[col for col in CLINICAL_COLS if col in clinical.columns]].copy()
    df = df.merge(clinical_use, on="caseid", how="left")
    df["sex_male"] = df["sex"].apply(encode_sex) if "sex" in df.columns else np.nan
    df["inadequate_sedation"] = (df["BIS"] > 60).astype(int)
    df["bis_zone"] = pd.cut(
        df["BIS"],
        bins=[-np.inf, 40, 60, np.inf],
        labels=["deep_BIS_lt_40", "adequate_40_60", "light_BIS_gt_60"],
    )
    return df


def main():
    ensure_dirs()
    clinical = load_clinical()
    caseids = vitaldb.find_cases(",".join(REQUIRED_TRACKS))
    caseids = sorted(int(caseid) for caseid in caseids)
    n_cases = min(int(os.getenv("N_CASES", DEFAULT_N_CASES)), len(caseids))
    selected_caseids = caseids[:n_cases]

    print("matching cases:", len(caseids))
    print("first 20:", caseids[:20])
    print("loading cases:", n_cases)

    raw = load_cases(selected_caseids, TRACKS)
    print("\nraw shape:", raw.shape)
    print("raw unique cases:", raw["caseid"].nunique())
    display(raw.head())

    raw_path = f"data/raw/vitaldb_raw_{n_cases}cases.csv"
    raw.to_csv(raw_path, index=False)

    df = build_fallback_dataset(raw)
    print("\nAfter fallback construction:")
    display(df.head())
    display(df.isna().mean().sort_values(ascending=False))

    df = df.sort_values(["caseid", "time_sec"]).copy()
    df[TS_COLS] = df.groupby("caseid")[TS_COLS].ffill(limit=10)

    print("\nAfter short forward fill:")
    display(df.isna().mean().sort_values(ascending=False))

    df = filter_signal_quality(df)
    df = crop_and_filter_cases(df)
    df = add_clinical_and_labels(df, clinical)

    clean_path = f"data/processed/vitaldb_clean_{n_cases}cases.csv"
    df.to_csv(clean_path, index=False)

    print("\nSaved raw:", raw_path)
    print("Saved clean:", clean_path)
    print("\n========== CLEAN DATA SUMMARY ==========")
    print("shape:", df.shape)
    print("unique cases:", df["caseid"].nunique())

    print("\nBIS describe:")
    display(df["BIS"].describe())

    print("\nBinary label distribution:")
    display(df["inadequate_sedation"].value_counts())
    display(df["inadequate_sedation"].value_counts(normalize=True))

    print("\nBIS zone distribution:")
    display(df["bis_zone"].value_counts())
    display(df["bis_zone"].value_counts(normalize=True))

    print("\nMissing rate:")
    display(df.isna().mean().sort_values(ascending=False).head(30))
    display(df.head())


if __name__ == "__main__":
    main()

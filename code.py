# ============================================================
# VitalDB 1차 데이터 로드: BIS + 기본 vital signs + clinical data
# ============================================================

import os
import re
import vitaldb
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

def display(obj):
    print(obj)

# ------------------------------------------------------------
# 0. 저장 폴더 만들기
# ------------------------------------------------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ------------------------------------------------------------
# 1. Clinical data 불러오기
# ------------------------------------------------------------
ALL_CASEIDS = list(range(1, 6389))

# vitaldb 1.7.2 returns an empty dataframe when caseids is omitted.
clinical = vitaldb.load_clinical_data(caseids=ALL_CASEIDS)

print("clinical shape:", clinical.shape)
display(clinical.head())

clinical.to_csv("data/raw/clinical.csv", index=False)

print("\n[Clinical columns related to demographics/surgery]")
demo_cols = [
    c for c in clinical.columns
    if any(k in c.lower() for k in ["age", "sex", "gender", "height", "weight", "bmi", "asa", "op", "ane"])
]
print(demo_cols)


# ------------------------------------------------------------
# 2. 전체 track name 불러오기
# ------------------------------------------------------------
track_names = vitaldb.get_track_names(caseids=clinical["caseid"].tolist())

def normalize_track_list(track_names):
    """
    vitaldb.get_track_names() 결과가 list일 수도 있고 DataFrame일 수도 있어서
    둘 다 처리하는 함수.
    """
    if isinstance(track_names, pd.DataFrame):
        # DataFrame 안의 모든 string 값을 flatten
        vals = []
        for col in track_names.columns:
            for item in track_names[col].dropna():
                if isinstance(item, (list, tuple, set)):
                    vals.extend(str(v) for v in item)
                else:
                    vals.append(str(item))
        # track name처럼 보이는 것만 남김: 보통 Device/Variable 형태
        vals = [v for v in vals if "/" in v]
        return sorted(set(vals))
    else:
        return sorted(set([str(x) for x in track_names if "/" in str(x)]))

all_tracks = normalize_track_list(track_names)

print("\nnumber of tracks:", len(all_tracks))
print("first 30 tracks:")
print(all_tracks[:30])


# ------------------------------------------------------------
# 3. track 검색 함수
# ------------------------------------------------------------
def search_tracks(keyword):
    keyword = keyword.lower()
    return [t for t in all_tracks if keyword in t.lower()]

def show_search(keyword):
    result = search_tracks(keyword)
    print(f"\n[{keyword}] {len(result)} tracks found")
    print(result[:50])
    return result


# 일단 주요 키워드 검색 출력
bis_candidates = show_search("BIS")
hr_candidates = show_search("HR")
mbp_candidates = show_search("MBP")
sbp_candidates = show_search("SBP")
dbp_candidates = show_search("DBP")
spo2_candidates = show_search("SPO2")
etco2_candidates = show_search("ETCO2")
ppf_candidates = show_search("PPF")
rftn_candidates = show_search("RFTN")


# ------------------------------------------------------------
# 4. 후보 track 자동 선택 함수
# ------------------------------------------------------------
def pick_track(possible_names, keyword=None, required=True):
    """
    possible_names 중 실제 all_tracks에 존재하는 첫 번째 track 선택.
    없으면 keyword 검색 결과 첫 번째 선택.
    그래도 없으면 None.
    """
    for name in possible_names:
        if name in all_tracks:
            return name
    
    if keyword is not None:
        found = search_tracks(keyword)
        if len(found) > 0:
            return found[0]
    
    if required:
        raise ValueError(f"Could not find track from {possible_names} / keyword={keyword}")
    return None


# ------------------------------------------------------------
# 5. 우리가 당장 쓸 기본 track 고르기
#    이름은 VitalDB 환경마다 약간 다를 수 있어서 후보를 여러 개 둠.
# ------------------------------------------------------------
selected = {}

selected["BIS"] = pick_track(
    ["BIS/BIS", "BIS/BIS_VALUE"],
    keyword="BIS",
    required=True
)

selected["HR"] = pick_track(
    ["Solar8000/HR", "Solar8000/PLETH_HR", "Solar8000/ECG_HR"],
    keyword="HR",
    required=False
)

selected["MBP"] = pick_track(
    ["Solar8000/ART_MBP", "Solar8000/NIBP_MBP", "Solar8000/MBP"],
    keyword="MBP",
    required=False
)

selected["SBP"] = pick_track(
    ["Solar8000/ART_SBP", "Solar8000/NIBP_SBP", "Solar8000/SBP"],
    keyword="SBP",
    required=False
)

selected["DBP"] = pick_track(
    ["Solar8000/ART_DBP", "Solar8000/NIBP_DBP", "Solar8000/DBP"],
    keyword="DBP",
    required=False
)

selected["SPO2"] = pick_track(
    ["Solar8000/PLETH_SPO2", "Solar8000/SPO2"],
    keyword="SPO2",
    required=False
)

selected["ETCO2"] = pick_track(
    ["Primus/ETCO2", "Solar8000/ETCO2", "Solar8000/VENT_ETCO2"],
    keyword="ETCO2",
    required=False
)

# 약물 관련 track: 일단 있으면 넣고, 없으면 건너뜀
selected["PPF_CE"] = pick_track(
    ["Orchestra/PPF20_CE", "Orchestra/PPF_CE", "Orchestra/PPF_CE_REMI"],
    keyword="PPF",
    required=False
)

selected["PPF_RATE"] = pick_track(
    ["Orchestra/PPF20_RATE", "Orchestra/PPF_RATE", "Orchestra/PPF20_VOL"],
    keyword="PPF",
    required=False
)

selected["RFTN_CE"] = pick_track(
    ["Orchestra/RFTN20_CE", "Orchestra/RFTN_CE"],
    keyword="RFTN",
    required=False
)

selected["RFTN_RATE"] = pick_track(
    ["Orchestra/RFTN20_RATE", "Orchestra/RFTN_RATE", "Orchestra/RFTN20_VOL"],
    keyword="RFTN",
    required=False
)

# None 제거
selected = {k: v for k, v in selected.items() if v is not None}

print("\n[Selected tracks]")
for k, v in selected.items():
    print(f"{k:10s}: {v}")

basic_tracks = list(selected.values())

print("\ntracks to load:")
print(basic_tracks)


# ------------------------------------------------------------
# 6. 해당 track들이 있는 case 찾기
# ------------------------------------------------------------
caseids = vitaldb.find_cases(",".join(basic_tracks))

print("\nnumber of matching cases:", len(caseids))
print("first 20 caseids:", caseids[:20])

# 만약 case가 너무 적으면 약물 track 때문에 조건이 빡센 것일 수 있음.
# 그럴 경우 BIS + 기본 vital만으로 다시 찾기.
if len(caseids) < 20:
    print("\nToo few cases. Retrying with only BIS + basic vital signs...")
    
    must_keys = ["BIS", "HR", "MBP", "SBP", "DBP", "SPO2", "ETCO2"]
    selected_basic = {k: v for k, v in selected.items() if k in must_keys}
    basic_tracks = list(selected_basic.values())
    
    print("\n[Basic selected tracks]")
    for k, v in selected_basic.items():
        print(f"{k:10s}: {v}")
    
    caseids = vitaldb.find_cases(",".join(basic_tracks))
    print("\nnumber of matching cases after retry:", len(caseids))
    print("first 20 caseids:", caseids[:20])


# ------------------------------------------------------------
# 7. 케이스 하나 불러와서 확인
# ------------------------------------------------------------
if len(caseids) == 0:
    raise RuntimeError("No matching cases found. Track 이름을 수동 확인해야 함.")

caseid = int(caseids[0])

arr = vitaldb.load_case(
    caseid,
    ",".join(basic_tracks),
    interval=1.0
)

df_one = pd.DataFrame(arr, columns=basic_tracks)
df_one["caseid"] = caseid
df_one["time_sec"] = np.arange(len(df_one))

print("\nLoaded one case:", caseid)
print("df_one shape:", df_one.shape)
display(df_one.head())

print("\nMissing rate in one case:")
display(df_one.isna().mean().sort_values(ascending=False))


# ------------------------------------------------------------
# 8. BIS column 찾고 그래프 그리기
# ------------------------------------------------------------
bis_track = selected["BIS"]

plt.figure(figsize=(12, 4))
plt.plot(df_one["time_sec"], df_one[bis_track])
plt.xlabel("Time (sec)")
plt.ylabel("BIS")
plt.title(f"BIS time series - case {caseid}")
bis_timeseries_path = f"data/processed/bis_timeseries_case_{caseid}.png"
plt.tight_layout()
plt.savefig(bis_timeseries_path, dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
df_one[bis_track].dropna().hist(bins=50)
plt.xlabel("BIS")
plt.ylabel("Count")
plt.title(f"BIS distribution - case {caseid}")
bis_distribution_path = f"data/processed/bis_distribution_case_{caseid}.png"
plt.tight_layout()
plt.savefig(bis_distribution_path, dpi=150)
plt.close()
print("\nSaved plot files:")
print(bis_timeseries_path)
print(bis_distribution_path)

print("\nBIS describe:")
display(df_one[bis_track].describe())


# ------------------------------------------------------------
# 9. 여러 케이스 불러오기
#    처음부터 전체 말고 30개만. 성공하면 나중에 100, 500으로 늘리기.
# ------------------------------------------------------------
N_CASES = min(30, len(caseids))

all_dfs = []

for cid in tqdm(caseids[:N_CASES], desc="Loading cases"):
    try:
        arr = vitaldb.load_case(
            int(cid),
            ",".join(basic_tracks),
            interval=1.0
        )
        temp = pd.DataFrame(arr, columns=basic_tracks)
        temp["caseid"] = int(cid)
        temp["time_sec"] = np.arange(len(temp))
        all_dfs.append(temp)
    except Exception as e:
        print(f"case {cid} failed:", e)

data = pd.concat(all_dfs, ignore_index=True)

print("\ncombined data shape:", data.shape)
display(data.head())

print("\nMissing rate in combined data:")
display(data.isna().mean().sort_values(ascending=False))


# ------------------------------------------------------------
# 10. BIS가 없는 row 제거 + label 생성
# ------------------------------------------------------------
data = data.dropna(subset=[bis_track]).copy()

# Wang 논문식 가장 기본 label: BIS > 60이면 inadequate sedation
data["inadequate_sedation"] = (data[bis_track] > 60).astype(int)

# 구간형 label도 만들어둠: 나중에 네 논문 차별화에 유용
# deep: BIS < 40
# adequate: 40 <= BIS <= 60
# light: BIS > 60
data["bis_zone"] = pd.cut(
    data[bis_track],
    bins=[-np.inf, 40, 60, np.inf],
    labels=["deep_BIS_lt_40", "adequate_40_60", "light_BIS_gt_60"]
)

print("\nBinary label distribution: inadequate_sedation")
display(data["inadequate_sedation"].value_counts(dropna=False))
display(data["inadequate_sedation"].value_counts(normalize=True, dropna=False))

print("\nBIS zone distribution:")
display(data["bis_zone"].value_counts(dropna=False))
display(data["bis_zone"].value_counts(normalize=True, dropna=False))


# ------------------------------------------------------------
# 11. 컬럼 이름을 분석하기 쉽게 rename
# ------------------------------------------------------------
rename_map = {v: k for k, v in selected.items() if v in data.columns}
data_renamed = data.rename(columns=rename_map)

print("\nrenamed columns:")
print(data_renamed.columns.tolist())


# ------------------------------------------------------------
# 12. 저장
# ------------------------------------------------------------
raw_path = f"data/processed/vitaldb_sample_{N_CASES}cases_raw_tracknames.csv"
renamed_path = f"data/processed/vitaldb_sample_{N_CASES}cases_renamed.csv"
selected_path = "data/processed/selected_tracks.csv"

data.to_csv(raw_path, index=False)
data_renamed.to_csv(renamed_path, index=False)

pd.DataFrame(
    [{"feature_name": k, "track_name": v} for k, v in selected.items()]
).to_csv(selected_path, index=False)

print("\nSaved files:")
print(raw_path)
print(renamed_path)
print(selected_path)


# ------------------------------------------------------------
# 13. 최종 sanity check
# ------------------------------------------------------------
print("\n========== SANITY CHECK ==========")
print("clinical:", clinical.shape)
print("number of matching cases:", len(caseids))
print("loaded cases:", N_CASES)
print("final data:", data_renamed.shape)
print("BIS column exists:", "BIS" in data_renamed.columns)
print("caseid unique:", data_renamed["caseid"].nunique())
print("BIS non-null:", data_renamed["BIS"].notna().sum() if "BIS" in data_renamed.columns else "No BIS column")
display(data_renamed.head())

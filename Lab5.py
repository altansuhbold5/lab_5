import pandas as pd
import numpy as np

raw = pd.read_csv("students.csv", dtype=str, keep_default_na=False)
print(raw.shape)
print(raw.head(10))

df = pd.read_csv("students.csv", na_values=["?", "N/A"])
print(df.shape)
print(df.dtypes)

miss_cnt = df.isna().sum()
missing_pct = (miss_cnt / len(df) * 100).round(1)
print(pd.DataFrame({"missing": miss_cnt, "missing_pct": missing_pct}))

before = df["city"].isna().sum()
df = df.replace(r"^\s*$", np.nan, regex=True)
after = df["city"].isna().sum()
print("өмнө:", before, "дараа:", after)

df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
miss_by_prog = df.groupby("program")["score_num"].apply(lambda s: s.isna().mean() * 100).round(1)
print(miss_by_prog)

df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
bad_type = df[df["age"].notna() & df["age_num"].isna()]
print(bad_type["age"].value_counts())

df["attendance_num"] = pd.to_numeric(df["attendance"], errors="coerce")
bad_age = df["age_num"].notna() & ~df["age_num"].between(16, 40)
bad_score = df["score_num"].notna() & ~df["score_num"].between(0, 100)
bad_att = df["attendance_num"].notna() & ~df["attendance_num"].between(0, 100)
print("age:", bad_age.sum(), "score:", bad_score.sum(), "attendance:", bad_att.sum())
print(df.loc[df["score_num"] > 100, ["id", "score", "program"]])

print(df["city"].value_counts(dropna=False))
df["city_clean"] = df["city"].str.strip().str.lower()
df["city_clean"] = df["city_clean"].replace({"уб": "улаанбаатар"})
print(df["city_clean"].value_counts(dropna=False))
print("өвөрмөц хот:", df["city_clean"].nunique())

d = df["enroll_date"]
d1 = pd.to_datetime(d, format="%Y/%m/%d", errors="coerce")
d2 = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
d3 = pd.to_datetime(d, format="%d/%m/%Y", errors="coerce")
df["enroll_dt"] = d1.fillna(d2).fillna(d3)
failed = df[df["enroll_date"].notna() & df["enroll_dt"].isna()]
print("амжилтгүй:", len(failed))
print(failed["enroll_date"].value_counts())

cols = ["id", "age", "score", "attendance", "city", "program", "enroll_date"]
n_dup = df.duplicated(subset=cols).sum()
print("илүү давхардал:", n_dup)
print(df[df.duplicated(subset=cols, keep=False)].sort_values("id").head(6))

dedup = df.drop_duplicates(subset=cols)
id_conflict = dedup[dedup.duplicated(subset="id", keep=False)].sort_values("id")
print("үлдсэн мөр:", len(dedup))
print("зөрчилтэй id:", id_conflict["id"].nunique())
print(id_conflict[["id", "score", "program"]])

def iqr_bounds(s, k=1.5):
    s = s.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

demo = pd.Series([18, 19, 20, 21, 22, 22, 23, 60])
lo, hi = iqr_bounds(demo)
print("demo:", lo, hi, demo[(demo < lo) | (demo > hi)].tolist())

ages = dedup["age_num"]
lo, hi = iqr_bounds(ages)
print("age bounds:", lo, hi)
print(ages[(ages < lo) | (ages > hi)].sort_values().tolist())

valid = dedup["score_num"][dedup["score_num"].between(0, 100)]
z = (valid - valid.mean()) / valid.std()
print(valid[z.abs() > 2.5].sort_values().tolist())

work = dedup.copy()
work["score_clean"] = work["score_num"].where(work["score_num"].between(0, 100))
print("нөхөхийн өмнө NaN:", work["score_clean"].isna().sum())
work["score_global"] = work["score_clean"].fillna(work["score_clean"].median())
grp_med = work.groupby("program")["score_clean"].transform("median")
work["score_group"] = work["score_clean"].fillna(grp_med)
print(work.groupby("program")[["score_clean", "score_global", "score_group"]].mean().round(1))

known = work.dropna(subset=["attendance_num", "score_clean"])
known = known[known["attendance_num"].between(0, 100)]
slope, intercept = np.polyfit(known["attendance_num"], known["score_clean"], 1)
print("slope:", round(slope, 3), "intercept:", round(intercept, 1))
print("corr:", round(known["attendance_num"].corr(known["score_clean"]), 2))
need = work["score_clean"].isna() & work["attendance_num"].between(0, 100)
work.loc[need, "score_reg"] = slope * work.loc[need, "attendance_num"] + intercept
print("таамагласан мөр:", need.sum())
print(work.loc[need, ["id", "attendance_num", "score_reg"]].head())

train = work.sample(frac=0.8, random_state=42)
test = work.drop(train.index)
train_med = train["score_clean"].median()
test_filled = test["score_clean"].fillna(train_med)
print("train:", len(train), "test:", len(test))
print("train медиан:", train_med, "бүх өгөгдлийн медиан:", work["score_clean"].median())
print("test NaN (өмнө -> дараа):", test["score_clean"].isna().sum(), "->", test_filled.isna().sum())

def clean_students(path):
    d = pd.read_csv(path, na_values=["?", "N/A"])
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    for c in ["age", "score", "attendance"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["age"] = d["age"].where(d["age"].between(16, 40))
    d["score"] = d["score"].where(d["score"].between(0, 100))
    d["attendance"] = d["attendance"].where(d["attendance"].between(0, 100))
    d["city"] = d["city"].str.strip().str.lower()
    d["city"] = d["city"].replace({"уб": "улаанбаатар"})
    d = d.drop_duplicates()
    return d

clean = clean_students("students.csv")
print("мөр:", len(raw), "->", len(clean))
print(clean.isna().sum())
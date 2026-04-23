import pandas as pd
import numpy as np

df = pd.read_pickle("multimodal_trials.pkl")

print("=== BASIC INFO ===")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("\nDtypes:")
print(df.dtypes)

print("\n=== LABEL / SUBJECT INFO ===")
if "label" in df.columns:
    print("label counts:")
    print(df["label"].value_counts(dropna=False))
if "subject_id" in df.columns:
    print("\nnum subjects:", df["subject_id"].nunique())
    print("trials per subject:")
    print(df.groupby("subject_id").size().describe())

print("\n=== MISSINGNESS ===")
print(df.isna().mean().sort_values(ascending=False))

def arr_len(x):
    return len(x) if isinstance(x, np.ndarray) else 0

def finite_frac(x):
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return 0.0
    x = np.asarray(x)
    return float(np.isfinite(x).mean())

for col in ["pupil", "gaze_x", "gaze_y"]:
    if col in df.columns:
        print(f"\n=== {col.upper()} STATS ===")
        lengths = df[col].apply(arr_len)
        print("length describe:")
        print(lengths.describe())
        finite = df[col].apply(finite_frac)
        print("finite fraction describe:")
        print(finite.describe())

print("\n=== EXAMPLE ROWS (metadata only) ===")
meta_cols = [c for c in ["subject_id", "trial", "label"] if c in df.columns]
print(df[meta_cols].head(10))

print("\n=== ONE EXAMPLE ROW WITH SMALL SLICES ===")
i = 0
row = df.iloc[i]
example = {}
for col in ["subject_id", "trial", "label"]:
    if col in df.columns:
        example[col] = row[col]
for col in ["pupil", "gaze_x", "gaze_y"]:
    if col in df.columns:
        arr = row[col]
        if isinstance(arr, np.ndarray):
            example[f"{col}_len"] = len(arr)
            example[f"{col}_first10"] = arr[:10]
        else:
            example[f"{col}_len"] = 0
            example[f"{col}_first10"] = None

for k, v in example.items():
    print(f"{k}: {v}")
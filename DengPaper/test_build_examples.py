from build_trial_examples import build_examples

df = build_examples("Pupil_dataset_py.mat")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nGroup counts:")
print(df["group_name"].value_counts())
print("\nUnique subjects:", df["subject_id"].nunique())
print("\nExample pupil length:", len(df.iloc[0]["pupil"]))
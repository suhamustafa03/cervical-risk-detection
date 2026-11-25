import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/21442367/images.csv"
OUTPUT = "data/phase2_split.csv"

df = pd.read_csv(INPUT)

# One label per patient
patients = (
    df[["patient_id", "patient_tz_type"]]
    .drop_duplicates("patient_id")
    .dropna(subset=["patient_id", "patient_tz_type"])
)

# 70% train, 30% temporary
train_p, temp_p = train_test_split(
    patients,
    test_size=0.30,
    stratify=patients["patient_tz_type"],
    random_state=42
)

# Split remaining 30% into 15% validation + 15% test
val_p, test_p = train_test_split(
    temp_p,
    test_size=0.50,
    stratify=temp_p["patient_tz_type"],
    random_state=42
)

train_ids = set(train_p["patient_id"])
val_ids = set(val_p["patient_id"])
test_ids = set(test_p["patient_id"])

def assign_split(pid):
    if pid in train_ids:
        return "train"
    if pid in val_ids:
        return "val"
    return "test"

df["split"] = df["patient_id"].map(assign_split)
df.to_csv(OUTPUT, index=False)

print("Patients:")
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))

print("\nImages:")
print(df["split"].value_counts())

print("\nPatient type distribution:")
print(pd.crosstab(
    df["split"],
    df["patient_tz_type"]
))

print("\nSaved:", OUTPUT)

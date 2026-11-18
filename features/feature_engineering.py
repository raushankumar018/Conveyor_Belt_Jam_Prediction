
import pandas as pd
import numpy as np
import os

# ---------------------------------------
# 1. Project folder paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

# ---------------------------------------
# 2. Load both dataset versions
# ---------------------------------------

v1 = pd.read_csv(
    os.path.join(DATA_DIR, "conveyor_v1.csv")
)

v2 = pd.read_csv(
    os.path.join(DATA_DIR, "conveyor_v2.csv")
)

v1["dataset_version"] = "V1"
v2["dataset_version"] = "V2"

print("Version 1 loaded:", v1.shape)
print("Version 2 loaded:", v2.shape)

# ---------------------------------------
# 3. Combine datasets
# ---------------------------------------

df = pd.concat([v1, v2], ignore_index=True)

print("\nCombined dataset:", df.shape)

# ---------------------------------------
# 4. Handle missing values
# ---------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

print("\nMissing values handled.")

# ---------------------------------------
# 5. Validate sensor measurements
# ---------------------------------------

if (df["rated_motor_current"] <= 0).any():
    raise ValueError("Rated motor current must be positive.")

if (df["motor_current"] < 0).any():
    raise ValueError("Motor current cannot be negative.")

if not df["next_shift_jam_status"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 or 1.")

# ---------------------------------------
# 6. Feature Engineering
# ---------------------------------------

# Feature 1: Motor Load Ratio

df["motor_load_ratio"] = (
    df["motor_current"] /
    df["rated_motor_current"]
)

# Feature 2: Belt Wear Pressure Index

df["belt_wear_pressure_index"] = (
    (df["belt_age_days"] / 60) *
    (1 + df["days_since_maintenance"] / 90) *
    (1 + df["dust_accumulation_index"] / 100)
)

print("\nNew features created successfully.")

# ---------------------------------------
# 7. Save engineered features
# ---------------------------------------

output_path = os.path.join(
    FEATURE_DIR,
    "engineered_features.csv"
)

df.to_csv(output_path, index=False)

# ---------------------------------------
# 8. Display results
# ---------------------------------------

print("\nFeature Engineering Completed!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nNew Feature Preview:")

print(
    df[
        [
            "motor_load_ratio",
            "belt_wear_pressure_index",
            "next_shift_jam_status"
        ]
    ].head()
)

print("\nDataset saved at:")
print(output_path)

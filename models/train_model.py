
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------
# 1. Define project paths
# ------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR, "features", "engineered_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------
# 2. Load engineered dataset
# ------------------------------------

df = pd.read_csv(FEATURE_FILE)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ------------------------------------
# 3. Select input features and target
# ------------------------------------

features = [
    "motor_current",
    "rated_motor_current",
    "vibration_level",
    "belt_age_days",
    "parcels_processed_per_minute",
    "dust_accumulation_index",
    "days_since_maintenance",
    "humidity",
    "motor_load_ratio",
    "belt_wear_pressure_index"
]

X = df[features]
y = df["next_shift_jam_status"]

# ------------------------------------
# 4. Split dataset
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------
# 5. Define ML models
# ------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
}

# ------------------------------------
# 6. Train and evaluate models
# ------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

# ------------------------------------
# 7. Compare models
# ------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by=["F1 Score", "Recall"],
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(MODEL_DIR, "model_comparison.csv"),
    index=False
)

# ------------------------------------
# 8. Select best model
# ------------------------------------

best_model_name = comparison.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# ------------------------------------
# 9. Save best model
# ------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(best_model, model_path)

# Save feature names for API usage
joblib.dump(
    features,
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

print("\nBest model saved successfully!")
print("Model path:", model_path)

print("\nMODEL TRAINING COMPLETED!")


import os
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------
# 1. Project paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR, "features", "engineered_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------
# 2. Configure MLflow
# ---------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db")
)

mlflow.set_experiment("Conveyor_Belt_Jam_Prediction")

MODEL_NAME = "ConveyorBeltJamModel"

# ---------------------------------------
# 3. Load dataset
# ---------------------------------------

df = pd.read_csv(DATA_FILE)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# 4. Define models
# ---------------------------------------

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

results = []

best_model = None
best_name = None
best_f1 = -1
best_recall = -1

# ---------------------------------------
# 5. Train and track experiments
# ---------------------------------------

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print("\nModel:", name)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", len(features))

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Normal", "Actual_Jam"],
            columns=["Predicted_Normal", "Predicted_Jam"]
        )

        cm_path = os.path.join(
            MODEL_DIR,
            name.replace(" ", "_") + "_confusion_matrix.csv"
        )

        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Score": f1
        })

        # Select best model using F1 and recall
        if (f1 > best_f1 or
            (f1 == best_f1 and recall > best_recall)):

            best_f1 = f1
            best_recall = recall
            best_model = model
            best_name = name

# ---------------------------------------
# 6. Register selected model
# ---------------------------------------

print("\nSelected Model:", best_name)

input_example = X_train.head(3)
signature = infer_signature(
    input_example,
    best_model.predict(input_example)
)

with mlflow.start_run(run_name="Best_Model_Registration"):

    mlflow.log_param("selected_model", best_name)
    mlflow.log_metric(
        "accuracy",
        accuracy_score(y_test, best_model.predict(X_test))
    )
    mlflow.log_metric("recall", best_recall)
    mlflow.log_metric("f1_score", best_f1)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME
    )

# ---------------------------------------
# 7. Save final comparison
# ---------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    os.path.join(MODEL_DIR, "mlflow_comparison.csv"),
    index=False
)

print("\n================================")
print("MLFLOW TRACKING COMPLETED!")
print("================================")
print("Selected Model:", best_name)
print("Registered Model:", MODEL_NAME)
print("F1 Score:", best_f1)
print("Recall:", best_recall)

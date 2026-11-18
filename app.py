"""FastAPI application will be implemented here."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# -----------------------------------
# 1. Initialize FastAPI
# -----------------------------------

app = FastAPI(
    title="Conveyor Belt Jam Prediction API",
    description="Predicts conveyor belt jamming during the next operating shift.",
    version="1.0"
)

# -----------------------------------
# 2. Load trained model
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "models", "feature_names.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
except Exception as error:
    raise RuntimeError(
        f"Could not load trained model: {error}"
    )

# -----------------------------------
# 3. Define sensor input
# -----------------------------------

class ConveyorInput(BaseModel):

    motor_current: float = Field(ge=0)
    rated_motor_current: float = Field(gt=0)

    vibration_level: float = Field(ge=0, le=20)
    belt_age_days: int = Field(ge=0, le=3650)

    parcels_processed_per_minute: float = Field(ge=0, le=500)

    dust_accumulation_index: float = Field(ge=0, le=100)

    days_since_maintenance: int = Field(ge=0, le=3650)

    humidity: float = Field(ge=0, le=100)


# -----------------------------------
# 4. Health endpoint
# -----------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": "Logistic Regression",
        "service": "Conveyor Belt Jam Prediction"
    }


# -----------------------------------
# 5. Prediction endpoint
# -----------------------------------

@app.post("/predict")
def predict(data: ConveyorInput):

    try:

        # Calculate engineered features

        motor_load_ratio = (
            data.motor_current /
            data.rated_motor_current
        )

        belt_wear_pressure_index = (
            (data.belt_age_days / 60) *
            (1 + data.days_since_maintenance / 90) *
            (1 + data.dust_accumulation_index / 100)
        )

        # Prepare model input

        input_data = pd.DataFrame([{

            "motor_current": data.motor_current,

            "rated_motor_current":
                data.rated_motor_current,

            "vibration_level":
                data.vibration_level,

            "belt_age_days":
                data.belt_age_days,

            "parcels_processed_per_minute":
                data.parcels_processed_per_minute,

            "dust_accumulation_index":
                data.dust_accumulation_index,

            "days_since_maintenance":
                data.days_since_maintenance,

            "humidity":
                data.humidity,

            "motor_load_ratio":
                motor_load_ratio,

            "belt_wear_pressure_index":
                belt_wear_pressure_index

        }])

        # Ensure correct feature order

        input_data = input_data[features]

        # Model prediction

        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            result = "Jam Risk"
        else:
            result = "Normal Operation"

        # Return response

        return {

            "prediction": result,

            "jam_status": prediction,

            "motor_load_ratio":
                round(motor_load_ratio, 3),

            "belt_wear_pressure_index":
                round(belt_wear_pressure_index, 3)

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

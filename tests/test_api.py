"""API tests will be added during implementation."""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    sensor_data = {
        "motor_current": 18.5,
        "rated_motor_current": 20,
        "vibration_level": 5.5,
        "belt_age_days": 55,
        "parcels_processed_per_minute": 75,
        "dust_accumulation_index": 85,
        "days_since_maintenance": 80,
        "humidity": 75
    }

    response = client.post("/predict", json=sensor_data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [
        "Jam Risk",
        "Normal Operation"
    ]

    assert result["jam_status"] in [0, 1]


def test_invalid_motor_current():
    sensor_data = {
        "motor_current": -10,
        "rated_motor_current": 20,
        "vibration_level": 5,
        "belt_age_days": 30,
        "parcels_processed_per_minute": 50,
        "dust_accumulation_index": 40,
        "days_since_maintenance": 20,
        "humidity": 60
    }

    response = client.post("/predict", json=sensor_data)

    assert response.status_code == 422


def test_invalid_rated_current():
    sensor_data = {
        "motor_current": 15,
        "rated_motor_current": 0,
        "vibration_level": 3,
        "belt_age_days": 20,
        "parcels_processed_per_minute": 40,
        "dust_accumulation_index": 30,
        "days_since_maintenance": 10,
        "humidity": 50
    }

    response = client.post("/predict", json=sensor_data)

    assert response.status_code == 422


def test_missing_sensor_input():
    sensor_data = {
        "motor_current": 15,
        "rated_motor_current": 20
    }

    response = client.post("/predict", json=sensor_data)

    assert response.status_code == 422

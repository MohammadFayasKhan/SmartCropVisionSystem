"""
predict.py
----------
Loads all trained model artifacts and exposes predict_from_sensors().
The model was trained on exactly 3 features: temperature, humidity, rainfall.

IMPORTANT : Binary Rain Sensor:
    The physical rain sensor (YL-83) only outputs a binary signal:
        rain = 1  → rain detected
        rain = 0  → no rain / dry

    The ML model was trained on rainfall in mm, so we map the binary
    signal to a representative mm value before scaling:
        rain = 1  →  120 mm  (active rain, mid-monsoon level)
        rain = 0  →   20 mm  (dry / no rainfall baseline)

    This mapping keeps the scaler input valid without retraining.
    The disease engine receives the raw binary value directly.
"""

import numpy as np
import pandas as pd
import joblib
import os

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load(fname):
    return joblib.load(os.path.join(BASE_DIR, fname))

# ── load all pipeline artifacts once at import time ────────────────────
model         = _load("crop_model.pkl")
scaler        = _load("scaler.pkl")
label_encoder = _load("label_encoder.pkl")
feature_names = _load("feature_names.pkl")   # ['temperature', 'humidity', 'rainfall']

# ── Binary → mm mapping for the trained model ──────────────────────────
RAIN_TO_MM = {1: 120.0, 0: 20.0}


def predict_from_sensors(temperature, humidity, soil_moisture, rain):
    """
    Run the full crop recommendation pipeline from sensor readings.

    Parameters
    ----------
    temperature   : float → °C from DHT11
    humidity      : float → % from DHT11
    soil_moisture : float → 0-100% from soil moisture sensor
    rain          : int   → 1 = rain detected, 0 = no rain (YL-83 digital output)

    Returns
    -------
    dict:
        recommended_crop : str
        confidence       : float (0-100)
        top3             : list of {"crop": str, "confidence": float}
        disease_alerts   : list of disease dicts from disease_engine
        features_used    : dict of sensor values for dashboard display
    """
    # ── Map binary rain to representative mm for the scaler ──────────
    rainfall_mm = RAIN_TO_MM.get(int(rain), 20.0)

    # ── Build feature row in trained column order ─────────────────────
    feature_row = pd.DataFrame(
        [[temperature, humidity, rainfall_mm]],
        columns=feature_names          # ['temperature', 'humidity', 'rainfall']
    )

    # ── Scale using training scaler ───────────────────────────────────
    feature_scaled = scaler.transform(feature_row)

    # ── Predict probabilities across all 22 crop classes ─────────────
    proba    = model.predict_proba(feature_scaled)[0]
    top3_idx = proba.argsort()[-3:][::-1]

    top3 = [
        {
            "crop"      : label_encoder.inverse_transform([int(i)])[0],
            "confidence": round(float(proba[i]) * 100, 1)
        }
        for i in top3_idx
    ]

    pred_idx         = int(proba.argmax())
    recommended_crop = label_encoder.inverse_transform([pred_idx])[0]
    confidence       = round(float(proba[pred_idx]) * 100, 1)

    # ── Disease risk detection (uses raw binary rain value) ──────────
    from disease_engine import detect_diseases
    disease_alerts = detect_diseases(temperature, humidity, rain, soil_moisture)

    # ── Risk flags for dashboard display ─────────────────────────────
    rain_int = int(rain)
    features_used = {
        "temperature"   : temperature,
        "humidity"      : humidity,
        "soil_moisture" : soil_moisture,
        "rain"          : rain_int,
        "fungal_risk"   : int(humidity > 80 and temperature > 28),
        "drought_risk"  : int(rain_int == 0 and soil_moisture < 25),
        "waterlog_risk" : int(rain_int == 1 or soil_moisture > 85),
    }

    return {
        "recommended_crop": recommended_crop,
        "confidence"      : confidence,
        "top3"            : top3,
        "disease_alerts"  : disease_alerts,
        "features_used"   : features_used,
    }

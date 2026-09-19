"""
api/app.py
================================================================
WHAT THIS FILE DOES:
This is the "Model Serving" stage of the MLOps lifecycle. It
exposes the trained TensorFlow model as a REST API using
FastAPI, so that other applications (a website, a mobile app,
another service, etc.) can send in Iris flower measurements and
get a prediction back.

WHY THIS MATTERS IN MLOPS:
A model sitting in a notebook or a saved file is not useful to
anyone. "Serving" is the step that turns a trained model into a
real, usable product. FastAPI is a popular choice because it is
fast, has automatic input validation (via Pydantic), and
generates interactive API documentation (Swagger UI) for free.
================================================================
"""

import json
import os

import joblib
import numpy as np
import tensorflow as tf
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------------------------------------------
# Load configuration so paths are not hard-coded in two places.
# ----------------------------------------------------------------
CONFIG_PATH = "config.yaml"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"[CONFIG ERROR] '{CONFIG_PATH}' not found. Run the API "
            "from the project root directory, e.g.:\n"
            "    uvicorn api.app:app --reload"
        )
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()

MODEL_PATH = config["paths"]["model_file"]
SCALER_PATH = config["paths"]["scaler_file"]
LABELS_PATH = config["paths"]["labels_file"]

app = FastAPI(
    title="Iris Flower Classification API",
    description=(
        "A simple MLOps demo API that serves a TensorFlow model "
        "trained on the classic Iris dataset. Part of a CCA "
        "academic MLOps pipeline project."
    ),
    version="1.0.0",
)

# ----------------------------------------------------------------
# Load the trained model, scaler, and label map ONCE at startup.
# Loading heavy artifacts per-request would be extremely slow.
# ----------------------------------------------------------------
model = None
scaler = None
class_names = None


@app.on_event("startup")
def load_artifacts():
    """
    Loads the TensorFlow model, the fitted StandardScaler, and the
    class label list from disk when the API server starts.

    Clear error messages are provided if any artifact is missing -
    this usually means `python src/train.py` has not been run yet.
    """
    global model, scaler, class_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"[MODEL LOADING ERROR] Could not find the trained model "
            f"at '{MODEL_PATH}'. Please train the model first by "
            f"running:\n    python src/train.py"
        )
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"[MODEL LOADING ERROR] Could not find the fitted scaler "
            f"at '{SCALER_PATH}'. Please train the model first by "
            f"running:\n    python src/train.py"
        )
    if not os.path.exists(LABELS_PATH):
        raise RuntimeError(
            f"[MODEL LOADING ERROR] Could not find the label map "
            f"at '{LABELS_PATH}'. Please train the model first by "
            f"running:\n    python src/train.py"
        )

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(LABELS_PATH, "r") as f:
            class_names = json.load(f)
        print("[STARTUP] Model, scaler, and labels loaded successfully.")
    except Exception as exc:
        # Wrapping the original exception keeps the traceback while
        # giving the developer an obvious, human-readable message.
        raise RuntimeError(
            f"[MODEL LOADING ERROR] Failed to load model artifacts: {exc}"
        ) from exc


# ----------------------------------------------------------------
# Pydantic schemas -> automatic request/response validation.
# ----------------------------------------------------------------
class IrisInput(BaseModel):
    """
    Defines and validates the four required Iris measurements.

    Pydantic automatically:
      - Rejects requests missing a field.
      - Rejects requests with the wrong data type (e.g. a string
        instead of a number).
      - Enforces the gt=0 constraint (measurements must be positive).
    """

    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }


class PredictionOutput(BaseModel):
    """Defines the shape of a successful prediction response."""

    prediction: str
    confidence: float
    probabilities: dict


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------
@app.get("/", tags=["Health"])
def read_root():
    """Simple health-check endpoint to confirm the API is running."""
    return {
        "message": "Iris Flower Classification API is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Reports whether the model artifacts are loaded and the service
    is ready to serve predictions. Useful for monitoring/uptime
    checks in a real deployment.
    """
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(payload: IrisInput):
    """
    Accepts four Iris flower measurements and returns the predicted
    species along with the model's confidence and full probability
    breakdown across all three classes.
    """
    if model is None or scaler is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Ensure 'python src/train.py' has "
                "been run and restart the API."
            ),
        )

    try:
        # Build the feature vector in the SAME order used during training.
        features = np.array(
            [
                [
                    payload.sepal_length,
                    payload.sepal_width,
                    payload.petal_length,
                    payload.petal_width,
                ]
            ]
        )

        # Apply the SAME scaler that was fitted during training - this
        # is critical. Using a different/unfitted scaler would silently
        # produce wrong predictions.
        scaled_features = scaler.transform(features)

        probabilities = model.predict(scaled_features, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        probability_map = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

        return PredictionOutput(
            prediction=predicted_class,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probability_map.items()},
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to generate a prediction: {exc}",
        )

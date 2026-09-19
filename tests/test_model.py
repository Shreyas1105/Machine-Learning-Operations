"""
tests/test_model.py
================================================================
WHAT THIS FILE DOES:
Automated tests that verify the pipeline actually works. These
run locally with `pytest` and automatically in GitHub Actions
CI on every push/pull request.

WHY THIS MATTERS IN MLOPS:
Automated testing is what makes CI/CD trustworthy. Without
tests, a CI pipeline can only tell you "the code imported
without crashing" - it can't tell you "the model still loads
and produces sane predictions". These tests cover:

  1. Model loading           - does the saved .keras file load?
  2. Prediction output shape - does the model output 3 class
                                probabilities that sum to ~1?
  3. API input validation    - does invalid input get rejected
                                with a 422 by Pydantic?
  4. API prediction response - does a valid request return a
                                well-formed prediction?
================================================================
"""

import json
import os

import joblib
import numpy as np
import pytest
import tensorflow as tf
import yaml
from fastapi.testclient import TestClient

# ----------------------------------------------------------------
# Locate the project root regardless of where pytest is invoked
# from, so paths in config.yaml resolve correctly.
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()
MODEL_PATH = os.path.join(PROJECT_ROOT, CONFIG["paths"]["model_file"])
SCALER_PATH = os.path.join(PROJECT_ROOT, CONFIG["paths"]["scaler_file"])
LABELS_PATH = os.path.join(PROJECT_ROOT, CONFIG["paths"]["labels_file"])

# Skip model-dependent tests gracefully (with a clear reason) if the
# model has not been trained yet, instead of failing with a confusing
# stack trace. This keeps the test suite friendly for a first-time
# clone where `python src/train.py` hasn't been run.
MODEL_ARTIFACTS_EXIST = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(SCALER_PATH)
    and os.path.exists(LABELS_PATH)
)

skip_if_no_model = pytest.mark.skipif(
    not MODEL_ARTIFACTS_EXIST,
    reason=(
        "Trained model artifacts not found. Run 'python src/train.py' "
        "before running these tests."
    ),
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded_model():
    """Loads the trained Keras model once for all tests in this module."""
    return tf.keras.models.load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def loaded_scaler():
    """Loads the fitted StandardScaler used during training."""
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def class_names():
    """Loads the ordered list of Iris class names."""
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def api_client():
    """
    Creates a FastAPI TestClient. Importing the app module triggers
    the startup event, which loads the model artifacts.
    """
    # Ensure the app reads config.yaml relative to the project root.
    os.chdir(PROJECT_ROOT)
    from api.app import app  # imported here so PROJECT_ROOT is set first

    with TestClient(app) as client:
        yield client


# ----------------------------------------------------------------
# 1. Model loading tests
# ----------------------------------------------------------------
@skip_if_no_model
def test_model_file_exists():
    """The trained model file should exist on disk after training."""
    assert os.path.exists(MODEL_PATH), (
        f"Expected trained model at {MODEL_PATH}. Run 'python src/train.py'."
    )


@skip_if_no_model
def test_model_loads_successfully(loaded_model):
    """The saved .keras model should load without errors."""
    assert loaded_model is not None
    assert isinstance(loaded_model, tf.keras.Model)


@skip_if_no_model
def test_model_expects_four_features(loaded_model):
    """The model's input layer should accept exactly 4 features."""
    input_shape = loaded_model.input_shape
    assert input_shape[-1] == 4, "Model should expect 4 Iris features."


# ----------------------------------------------------------------
# 2. Prediction output tests
# ----------------------------------------------------------------
@skip_if_no_model
def test_prediction_output_shape(loaded_model, loaded_scaler):
    """Predictions should return one probability per class (3 total)."""
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # a classic setosa example
    scaled_sample = loaded_scaler.transform(sample)
    predictions = loaded_model.predict(scaled_sample, verbose=0)

    assert predictions.shape == (1, 3), "Expected 3 class probabilities."


@skip_if_no_model
def test_prediction_probabilities_sum_to_one(loaded_model, loaded_scaler):
    """Softmax output should sum to (approximately) 1.0."""
    sample = np.array([[6.7, 3.1, 4.7, 1.5]])  # a classic versicolor example
    scaled_sample = loaded_scaler.transform(sample)
    predictions = loaded_model.predict(scaled_sample, verbose=0)[0]

    assert pytest.approx(1.0, abs=1e-3) == float(np.sum(predictions))


@skip_if_no_model
def test_known_setosa_sample_predicts_setosa(loaded_model, loaded_scaler, class_names):
    """
    A textbook-typical setosa sample should be classified as setosa.
    This is a simple 'sanity check' test - it doesn't guarantee the
    model is perfect, but it catches obviously broken models.
    """
    sample = np.array([[5.0, 3.6, 1.4, 0.2]])
    scaled_sample = loaded_scaler.transform(sample)
    prediction = loaded_model.predict(scaled_sample, verbose=0)[0]
    predicted_class = class_names[int(np.argmax(prediction))]

    assert predicted_class == "setosa"


# ----------------------------------------------------------------
# 3. API input validation tests
# ----------------------------------------------------------------
@skip_if_no_model
def test_api_rejects_missing_field(api_client):
    """A request missing a required field should return HTTP 422."""
    incomplete_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        # petal_width intentionally missing
    }
    response = api_client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


@skip_if_no_model
def test_api_rejects_non_numeric_field(api_client):
    """A request with a non-numeric value should return HTTP 422."""
    bad_payload = {
        "sepal_length": "not_a_number",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


@skip_if_no_model
def test_api_rejects_negative_value(api_client):
    """Negative measurements are physically impossible and should be rejected."""
    bad_payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ----------------------------------------------------------------
# 4. API prediction response tests
# ----------------------------------------------------------------
@skip_if_no_model
def test_api_valid_prediction_response(api_client):
    """A valid request should return a well-formed prediction response."""
    valid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = api_client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert body["prediction"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= body["confidence"] <= 1.0


@skip_if_no_model
def test_api_health_check(api_client):
    """The /health endpoint should report the model as loaded."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

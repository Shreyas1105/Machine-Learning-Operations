"""
train.py
================================================================
WHAT THIS FILE DOES:
This is the heart of the "Model Development" and "Experiment
Tracking" stages of the MLOps lifecycle. It:

    1. Loads the Iris dataset.
    2. Splits it into train / validation / test sets.
    3. Scales the features (important for neural networks).
    4. Builds a simple feed-forward neural network with
       tf.keras.
    5. Trains the model.
    6. Evaluates it on the held-out test set.
    7. Logs every important detail (hyperparameters, metrics,
       the model itself) to MLflow so the experiment is
       reproducible and comparable to future runs.
    8. Registers the trained model into the MLflow Model
       Registry - this is the "Governance" stage, where a
       model officially becomes a versioned, trackable asset.
    9. Saves the trained model + scaler + label map to disk so
       the FastAPI service (api/app.py) can load and serve it.

WHY THIS MATTERS IN MLOPS:
Without experiment tracking, every time you retrain a model you
lose the history of "what settings gave what result". MLflow
solves this by recording every run automatically. The Model
Registry then gives you a single source of truth for "which
model version is currently the official one".
================================================================
"""

import json
import os

import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Reads config.yaml so that hyperparameters are NOT hard-coded.

    MLOps reason: configuration-as-code makes every run
    reproducible and lets us change an experiment without
    touching the training logic.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[CONFIG ERROR] Could not find '{config_path}'. "
            "Make sure you run this script from the project root "
            "directory (MLOps-CCA/), e.g.:  python src/train.py"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(cfg: dict):
    """
    Loads the Iris dataset and splits it into train/val/test.

    We do a two-step split:
        1. Split off the TEST set first (never touched again
           until final evaluation).
        2. Split the remaining data into TRAIN and VALIDATION.

    This mirrors real-world MLOps practice: the test set acts
    as a stand-in for "unseen production data".
    """
    iris = load_iris()
    X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
    y = iris.target  # 3 classes: setosa, versicolor, virginica
    class_names = list(iris.target_names)

    seed = cfg["data"]["random_seed"]
    test_split = cfg["data"]["test_split"]
    val_split = cfg["data"]["val_split"]

    # Step 1: carve out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    # Step 2: carve out validation from what remains
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=seed, stratify=y_temp
    )

    # --- Feature scaling ---
    # Neural networks train faster and more reliably when inputs
    # are on a similar scale. We FIT the scaler only on the
    # training data (to avoid "data leakage" from val/test sets)
    # and then apply (transform) it to all three splits.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        class_names,
    )


def build_model(cfg: dict, input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Builds a simple, explainable feed-forward neural network.

    Architecture:
        Input (4 features)
          -> Dense hidden layer 1 (ReLU)
          -> Dense hidden layer 2 (ReLU)
          -> Dense output layer (Softmax, 3 classes)

    We keep this intentionally small - this is an academic
    project, and a deep/complex network is unnecessary for a
    dataset as simple as Iris. Simplicity also makes it easy to
    explain during a viva.
    """
    m_cfg = cfg["model"]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_features"),
            tf.keras.layers.Dense(
                m_cfg["hidden_layer_1_units"],
                activation=m_cfg["activation"],
                name="hidden_layer_1",
            ),
            tf.keras.layers.Dense(
                m_cfg["hidden_layer_2_units"],
                activation=m_cfg["activation"],
                name="hidden_layer_2",
            ),
            # Softmax output -> gives a probability for each of the 3 classes
            tf.keras.layers.Dense(
                num_classes,
                activation=m_cfg["output_activation"],
                name="output_layer",
            ),
        ],
        name="iris_classifier",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg["training"]["learning_rate"]
    )

    # sparse_categorical_crossentropy is used because our labels
    # (y) are plain integers (0, 1, 2) rather than one-hot vectors.
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("=" * 60)
    print("MLOps Iris Pipeline - Training & Experiment Tracking")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    cfg = load_config("config.yaml")

    # ---------------------------------------------------------
    # 2. Prepare data
    # ---------------------------------------------------------
    print("\n[1/6] Loading and preparing data...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        class_names,
    ) = load_and_prepare_data(cfg)
    print(
        f"      Train samples: {len(X_train)} | "
        f"Val samples: {len(X_val)} | Test samples: {len(X_test)}"
    )

    # ---------------------------------------------------------
    # 3. Configure MLflow
    # ---------------------------------------------------------
    print("\n[2/6] Configuring MLflow experiment tracking...")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Enabling autolog gives us free tracking of TensorFlow
    # training metrics per epoch, but we ALSO log explicit
    # params/metrics below so the run is transparent and easy
    # to explain in a viva (not "magic").
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name="iris_tf_training") as run:
        run_id = run.info.run_id
        print(f"      MLflow run started: {run_id}")

        # -----------------------------------------------------
        # 4. Log hyperparameters (BEFORE training)
        # -----------------------------------------------------
        mlflow.log_param("learning_rate", cfg["training"]["learning_rate"])
        mlflow.log_param("batch_size", cfg["training"]["batch_size"])
        mlflow.log_param("epochs", cfg["training"]["epochs"])
        mlflow.log_param(
            "hidden_layer_1_units", cfg["model"]["hidden_layer_1_units"]
        )
        mlflow.log_param(
            "hidden_layer_2_units", cfg["model"]["hidden_layer_2_units"]
        )
        mlflow.log_param("activation", cfg["model"]["activation"])
        mlflow.log_param("random_seed", cfg["data"]["random_seed"])
        mlflow.log_param("test_split", cfg["data"]["test_split"])
        mlflow.log_param("val_split", cfg["data"]["val_split"])

        # -----------------------------------------------------
        # 5. Build and train the model
        # -----------------------------------------------------
        print("\n[3/6] Building the TensorFlow model...")
        model = build_model(cfg, input_dim=X_train.shape[1], num_classes=3)
        model.summary()

        # Log a text description of the architecture as an
        # MLflow artifact - useful documentation for governance.
        arch_summary_lines = []
        model.summary(print_fn=lambda line: arch_summary_lines.append(line))
        arch_path = "model_architecture.txt"
        with open(arch_path, "w") as f:
            f.write("\n".join(arch_summary_lines))
        mlflow.log_artifact(arch_path)
        os.remove(arch_path)

        print("\n[4/6] Training the model...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            verbose=2,
        )

        # -----------------------------------------------------
        # 6. Evaluate on the untouched test set
        # -----------------------------------------------------
        print("\n[5/6] Evaluating on the test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        final_train_accuracy = history.history["accuracy"][-1]
        final_val_accuracy = history.history["val_accuracy"][-1]
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        print(f"      Final Train Accuracy: {final_train_accuracy:.4f}")
        print(f"      Final Val Accuracy:   {final_val_accuracy:.4f}")
        print(f"      Test Accuracy:        {test_accuracy:.4f}")
        print(f"      Test Loss:            {test_loss:.4f}")

        # Log the metrics that matter most for this project.
        mlflow.log_metric("final_train_accuracy", final_train_accuracy)
        mlflow.log_metric("final_val_accuracy", final_val_accuracy)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        # A simple extra evaluation metric: per-class accuracy report,
        # saved as a text artifact for governance/documentation.
        from sklearn.metrics import classification_report

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        report = classification_report(y_test, y_pred, target_names=class_names)
        print("\n      Classification Report:\n")
        print(report)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # -----------------------------------------------------
        # 7. Save model + scaler + label map to disk
        #    (used later by the FastAPI service)
        # -----------------------------------------------------
        print("\n[6/6] Saving model artifacts to disk and MLflow...")
        os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)

        model.save(cfg["paths"]["model_file"])
        joblib.dump(scaler, cfg["paths"]["scaler_file"])
        with open(cfg["paths"]["labels_file"], "w") as f:
            json.dump(class_names, f)

        print(f"      Model saved to:  {cfg['paths']['model_file']}")
        print(f"      Scaler saved to: {cfg['paths']['scaler_file']}")
        print(f"      Labels saved to: {cfg['paths']['labels_file']}")

        # -----------------------------------------------------
        # 8. Log the model to MLflow + Register it
        #    (This is the "Governance" step of the lifecycle)
        # -----------------------------------------------------
        # mlflow.tensorflow.log_model both stores the model as an
        # MLflow artifact AND (via registered_model_name) creates/
        # updates an entry in the MLflow Model Registry. Every time
        # this script runs, a NEW version of the registered model
        # is created - giving us full version history.
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )

        print(
            f"\n      Model registered in MLflow Model Registry as "
            f"'{cfg['mlflow']['registered_model_name']}'."
        )
        print(f"\nMLflow run ID: {run_id}")
        print(
            "To view this run, start the MLflow UI from the project "
            "root with:\n    mlflow ui --backend-store-uri mlruns"
        )

    # ---------------------------------------------------------
    # 9. Also export small reference/current CSVs for Evidently
    #    monitoring demo (uses the same scaled test features).
    # ---------------------------------------------------------
    _export_monitoring_datasets(cfg, X_train, X_test, y_train, y_test)

    print("\nTraining complete.")


def _export_monitoring_datasets(cfg, X_train, X_test, y_train, y_test):
    """
    Creates two small CSV files used later by monitoring/monitor.py:

      - reference_data.csv: represents "known good" data the model
        was trained on (a stand-in for production data at launch time).
      - current_data.csv: represents "new" incoming data that we want
        to check for drift (here, the untouched test split acts as
        a simple stand-in for this in an academic setting).

    MLOps reason: monitoring tools like Evidently compare a
    reference dataset against current/live data to catch
    data drift before it silently degrades model performance.
    """
    feature_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    reference_df = pd.DataFrame(X_train, columns=feature_names)
    reference_df["target"] = y_train

    current_df = pd.DataFrame(X_test, columns=feature_names)
    current_df["target"] = y_test

    os.makedirs(cfg["paths"].get("reference_data", "data").rsplit("/", 1)[0], exist_ok=True)
    reference_df.to_csv(cfg["paths"]["reference_data"], index=False)
    current_df.to_csv(cfg["paths"]["current_data"], index=False)

    print(
        f"\n      Monitoring datasets exported:\n"
        f"        Reference -> {cfg['paths']['reference_data']}\n"
        f"        Current   -> {cfg['paths']['current_data']}"
    )


if __name__ == "__main__":
    main()

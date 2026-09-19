# MLOps Pipeline for Iris Flower Classification Using TensorFlow

A small, complete, end-to-end MLOps project built for a CCA (Continuous
Comprehensive Assessment) academic submission. It demonstrates the full
machine-learning lifecycle — from model development to CI/CD, model
serving, monitoring, and governance — using the classic Iris dataset.

> **Academic note:** This project is intentionally kept simple. Every file
> is commented to explain *what* the code does and *why* it matters in an
> MLOps context, so it is easy to explain during a viva.

---

## Table of Contents

- [A. Project Overview](#a-project-overview)
- [B. MLOps Lifecycle](#b-mlops-lifecycle)
- [C. Architecture](#c-architecture)
- [D. Technologies Used](#d-technologies-used)
- [E. Project Structure](#e-project-structure)
- [F. Installation](#f-installation)
- [G. Virtual Environment Setup (Windows PowerShell)](#g-virtual-environment-setup-windows-powershell)
- [G2. Running the Project in VS Code](#g2-running-the-project-in-vs-code)
- [H. Model Training](#h-model-training)
- [I. MLflow Experiment Tracking](#i-mlflow-experiment-tracking)
- [J. MLflow Model Registry](#j-mlflow-model-registry)
- [K. Running FastAPI](#k-running-fastapi)
- [L. Testing the API](#l-testing-the-api)
- [M. Running Pytest](#m-running-pytest)
- [N. Running Evidently Monitoring](#n-running-evidently-monitoring)
- [O. GitHub Actions](#o-github-actions)
- [P. Git Commands](#p-git-commands)
- [Q. Troubleshooting](#q-troubleshooting)
- [R. How All the Tools Work Together](#r-how-all-the-tools-work-together)

---

## A. Project Overview

This project trains a neural network to classify Iris flowers into one of
three species — *setosa*, *versicolor*, or *virginica* — based on four
measurements: sepal length, sepal width, petal length, and petal width.

But the real point of the project isn't the model itself (Iris
classification is a "hello world" problem in ML). The point is
**everything around the model** — the practices and tools that turn a
one-off training script into a maintainable, trustworthy, production-style
system. That collection of practices is called **MLOps** (Machine Learning
Operations).

---

## B. MLOps Lifecycle

This project implements the following stages, each mapped to a real tool:

| Stage | Tool | What it does |
|---|---|---|
| Data preparation | scikit-learn | Loads Iris data, splits train/val/test, scales features |
| Model development | TensorFlow (`tf.keras`) | Defines and trains a feed-forward neural network |
| Experiment tracking | MLflow Tracking | Logs hyperparameters, metrics, and artifacts for every run |
| Version control | Git + GitHub | Tracks all code changes and enables collaboration |
| CI/CD | GitHub Actions | Automatically trains + tests the project on every push |
| Governance | MLflow Model Registry | Versions the trained model and tracks its lifecycle |
| Model serving | FastAPI | Exposes the trained model as a REST API |
| Monitoring | Evidently AI | Detects data drift/quality issues in production-like data |

---

## C. Architecture

```
Data
  ↓
Data Preparation           (src/train.py -> load, split, scale)
  ↓
TensorFlow Model Development (src/train.py -> tf.keras model)
  ↓
Model Training & Evaluation  (src/train.py -> fit + evaluate)
  ↓
MLflow Experiment Tracking   (src/train.py -> mlflow.log_param/metric)
  ↓
Git/GitHub                   (version control of all code)
  ↓
GitHub Actions CI            (.github/workflows/test.yml)
  ↓
MLflow Model Registry        (src/train.py -> mlflow.tensorflow.log_model)
  ↓
FastAPI Model Serving        (api/app.py -> POST /predict)
  ↓
Evidently Monitoring         (monitoring/monitor.py -> drift report)
  ↓
Feedback / Model Improvement (insights feed back into retraining)
```

Each arrow represents a real, working handoff in this repository — not
just documentation. Running the commands in this README executes every
stage for real.

---

## D. Technologies Used

- **Python 3.10** — programming language
- **TensorFlow 2.21.0** — model development (`tf.keras`)
- **scikit-learn** — dataset loading, train/val/test split, feature scaling
- **MLflow** — experiment tracking + model registry
- **FastAPI** + **Uvicorn** — model serving as a REST API
- **Pydantic** — request validation for the API
- **Evidently AI** — data drift and data quality monitoring
- **GitHub Actions** — CI/CD automation
- **pytest** — automated testing
- **PyYAML** — reading `config.yaml`

---

## E. Project Structure

```
MLOps-CCA/
│
├── .github/
│   └── workflows/
│       └── test.yml          # CI pipeline: install deps, train, test
│
├── src/
│   └── train.py               # Data prep, model training, MLflow tracking + registry
│
├── model/                     # Saved model, scaler, and label map (generated)
│
├── tests/
│   └── test_model.py          # pytest suite: model + API tests
│
├── api/
│   └── app.py                 # FastAPI app: /predict endpoint
│
├── monitoring/
│   └── monitor.py             # Evidently AI drift/quality report generator
│
├── data/                      # reference_data.csv / current_data.csv (generated)
│
├── requirements.txt
├── README.md
├── .gitignore
└── config.yaml                # All tunable parameters (no hard-coding)
```

---

## F. Installation

### Prerequisites

- Windows 10/11
- Python **3.10** installed and available on PATH
- Git installed
- (Optional but recommended) VS Code

> **Why Python 3.10 specifically?** TensorFlow, MLflow, and Evidently each
> move at different speeds, and their supported dependency ranges (numpy,
> protobuf, etc.) don't line up cleanly on very new Python releases like
> 3.12/3.13 yet — you can end up with `pip` unable to resolve a working
> combination at all. Python 3.10 is old enough that every library in this
> project has mature, well-tested wheels that all work together, which
> matters more for a reproducible academic project than having the newest
> Python version.

If you already have a newer Python version installed (e.g. 3.13) for other
work, you don't need to remove it. Install Python 3.10 from
[python.org](https://www.python.org/downloads/release/python-31011/)
alongside it, then use the Python Launcher to target it specifically when
creating the virtual environment (see Section G below).

Check what's on your PATH in PowerShell:

```powershell
python --version
py -0    # lists every Python version the launcher can see
```

### Clone the repository

```powershell
git clone https://github.com/<your-username>/MLOps-CCA.git
cd MLOps-CCA
```

---

## G. Virtual Environment Setup (Windows PowerShell)

Using a virtual environment keeps this project's dependencies isolated
from the rest of your system.

```powershell
# 1. Create the virtual environment using Python 3.10 specifically
#    (the -3.10 flag tells the Python Launcher which version to use,
#    even if a different version like 3.13 is your system default)
py -3.10 -m venv venv

# If the `py` launcher isn't available, and `python --version` already
# reports 3.10.x, you can use this instead:
# python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# If PowerShell blocks the activation script with an execution-policy
# error, run this once (in an elevated PowerShell) and try again:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3. Confirm the venv is actually using Python 3.10
python --version

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all project dependencies
pip install -r requirements.txt
```

You should see `(venv)` at the start of your PowerShell prompt once the
environment is active.

---

## G2. Running the Project in VS Code

You can do everything below either in a plain PowerShell terminal (Sections
H onward) or directly inside VS Code, which is generally more convenient
since you get IntelliSense, inline debugging, and one place for your
terminal and editor. Here's the full setup:

### 1. Open the project folder

`File > Open Folder...` and select the `MLOps-CCA` folder itself (not its
parent) — this makes `MLOps-CCA` your workspace root, which matters
because every script in this project expects to be run from there (that's
where `config.yaml` lives).

### 2. Install the Python extension

If you don't already have it: open the Extensions panel (`Ctrl+Shift+X`),
search for **"Python"** (by Microsoft), and install it. This gives you
the interpreter picker, linting, debugging, and the `.venv`/`venv`
auto-detection used below.

### 3. Create the virtual environment (if you haven't already)

Open a terminal inside VS Code with `` Ctrl+` `` (or `Terminal > New
Terminal`) — this opens a PowerShell terminal already rooted in your
project folder — and run the same commands as Section G:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Point VS Code at the venv's Python interpreter

This is the step people most often miss, and it's why code can run fine
in the terminal but show red squiggly "module not found" warnings in the
editor, or why the wrong Python version gets used when you click "Run".

- Press `Ctrl+Shift+P` to open the Command Palette
- Type and select **"Python: Select Interpreter"**
- Choose the one that points into your project's `venv` folder, e.g.
  `.\venv\Scripts\python.exe` (VS Code usually labels it
  `('venv': venv)` and shows the Python version — confirm it says 3.10.x)

Once selected, the interpreter shown in VS Code's bottom status bar should
say something like `Python 3.10.x ('venv')`. Every terminal you open from
now on inside VS Code will automatically activate this same venv.

### 5. Run each stage

With the venv selected, open a VS Code terminal and run these exactly as
in the rest of this README — nothing changes about the commands
themselves, VS Code is just a nicer place to run and read them:

```powershell
# Train the model
python src/train.py

# View MLflow UI (then Ctrl+click the localhost link VS Code prints)
mlflow ui --backend-store-uri mlruns

# Serve the API (open a second terminal with the + icon so training/UI
# processes stay running in the first one)
uvicorn api.app:app --reload

# Run tests
pytest tests/ -v

# Run monitoring
python monitoring/monitor.py
```

VS Code automatically detects `http://127.0.0.1:...` links printed in the
terminal and makes them clickable — use those to open the MLflow UI,
Swagger docs, or the Evidently HTML report (VS Code will offer to open
HTML files in your default browser, or you can right-click the file in
the Explorer panel and choose "Reveal in File Explorer").

### 6. (Optional) Debugging `src/train.py` or `api/app.py` with breakpoints

If you want to step through the code line-by-line instead of just running
it:

- Click in the left margin next to a line number to set a breakpoint (a
  red dot appears)
- Open the file you want to debug (e.g. `src/train.py`)
- Press `F5`, or click the "Run and Debug" icon in the sidebar and choose
  **"Python File"** — this runs the currently open file under the debugger
  using your selected venv interpreter
- Execution will pause at your breakpoint, where you can inspect variables
  (e.g. `X_train.shape`, `history.history`) in the "Variables" panel

For `api/app.py` specifically, since it needs to be launched via `uvicorn`
rather than run directly, it's usually simpler to just run
`uvicorn api.app:app --reload` in the terminal and set breakpoints inside
`predict()` — VS Code's debugger will still attach if you start it via
`Run > Start Debugging` with a small `.vscode/launch.json` configuration
targeting `uvicorn`, but for an academic project this is optional; the
terminal-based workflow above is enough to demonstrate and screenshot
everything you need.

---

## H. Model Training

Run the training script from the **project root** (the `MLOps-CCA/`
folder):

```powershell
python src/train.py
```

This will:

1. Load and split the Iris dataset (train / validation / test).
2. Scale the features.
3. Build and train a small feed-forward neural network.
4. Print training/validation/test accuracy and a classification report.
5. Save the model to `model/iris_model.keras`, the scaler to
   `model/scaler.pkl`, and the class labels to `model/label_classes.json`.
6. Log everything (hyperparameters, metrics, artifacts) to MLflow.
7. Register the trained model in the MLflow Model Registry.
8. Export `data/reference_data.csv` and `data/current_data.csv` for the
   Evidently monitoring step.

**Every run above updates a live MLflow run and creates a new registered
model version — nothing here is pre-generated or faked.**

---

## I. MLflow Experiment Tracking

MLflow stores tracking data locally in the `mlruns/` folder (created
automatically, ignored by Git).

To view your experiments in a browser:

```powershell
mlflow ui --backend-store-uri mlruns
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

In the UI you can:

- See every training run under the `iris_classification_experiment`
  experiment.
- Compare hyperparameters (learning rate, batch size, epochs) across runs.
- View logged metrics: `final_train_accuracy`, `final_val_accuracy`,
  `test_accuracy`, `final_train_loss`, `final_val_loss`, `test_loss`.
- Download logged artifacts: the model architecture summary and the
  classification report.

Run `python src/train.py` multiple times (e.g. after changing
`config.yaml`) to generate multiple comparable runs — this is a great
screenshot for your CCA report.

---

## J. MLflow Model Registry

Model registration happens automatically inside `src/train.py` via:

```python
mlflow.tensorflow.log_model(
    model=model,
    artifact_path="model",
    registered_model_name=cfg["mlflow"]["registered_model_name"],
)
```

**Training → Registration workflow:**

1. `src/train.py` trains a model inside an MLflow run.
2. `mlflow.tensorflow.log_model(..., registered_model_name=...)` both logs
   the model as a run artifact **and** registers it under the name
   `iris_tf_classifier` in the Model Registry.
3. Each time you re-run training, a **new version** (Version 1, Version 2,
   ...) is added under the same registered model name — nothing is
   overwritten, so you always have a full history.
4. In the MLflow UI, open the **"Models"** tab to see all versions of
   `iris_tf_classifier`, their creation timestamps, and which run produced
   each one.

**Simple governance workflow (documented, not automated, to keep this
academic project easy to explain):**

- A model version starts in an implicit "None" stage right after
  registration.
- After you (the "model owner") review its metrics in the MLflow UI, you
  can manually promote a version to **"Staging"** and later **"Production"**
  using the MLflow UI or the `MlflowClient.transition_model_version_stage`
  API.
- This manual review step is a simple stand-in for real-world model
  governance: a human always checks metrics before a model is trusted with
  live traffic.

---

## K. Running FastAPI

Make sure you have trained a model first (Section H), then from the
project root:

```powershell
uvicorn api.app:app --reload
```

The API will be available at:

- Root/health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The Swagger UI lets you try the `/predict` endpoint directly from your
browser — no extra tools needed.

---

## L. Testing the API

### Option 1: Swagger UI

Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand
`POST /predict`, click **"Try it out"**, and submit the example payload.

### Option 2: PowerShell (`curl`/`Invoke-RestMethod`)

```powershell
$body = @{
    sepal_length = 5.1
    sepal_width  = 3.5
    petal_length = 1.4
    petal_width  = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

Expected response shape:

```json
{
  "prediction": "setosa",
  "confidence": 0.98,
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.015,
    "virginica": 0.005
  }
}
```

---

## M. Running Pytest

From the project root (with the virtual environment active and a model
already trained):

```powershell
pytest tests/ -v
```

This runs tests for:

- Model file existence and successful loading
- Correct input/output shapes and probability sums
- API rejection of missing/invalid/negative inputs (HTTP 422)
- API returning a well-formed prediction for valid input (HTTP 200)

If no trained model exists yet, model-dependent tests are automatically
**skipped** (not failed) with a clear message telling you to run
`python src/train.py` first.

---

## N. Running Evidently Monitoring

After training (Section H) has generated `data/reference_data.csv` and
`data/current_data.csv`, run:

```powershell
python monitoring/monitor.py
```

This produces `monitoring/monitoring_report.html`. Open it in any browser
to see:

- **Data Drift** results: whether each feature's distribution has shifted
  between the reference (training) data and the current (test) data.
- **Data Quality** results: missing values, duplicates, and basic
  statistics for both datasets.

This HTML report is a great screenshot for your CCA report and clearly
demonstrates real monitoring output (not a mock dashboard).

---

## O. GitHub Actions

The workflow at `.github/workflows/test.yml` runs automatically on every
`push` and `pull_request`:

1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies from `requirements.txt`.
4. Trains the model (so fresh artifacts exist to test against).
5. Runs `pytest tests/ -v`.
6. Shows a green check (success) or red X (failure) on GitHub.

To see it in action:

```powershell
git add .
git commit -m "Trigger CI run"
git push
```

Then open your repository on GitHub and click the **"Actions"** tab to
watch the workflow run and view its logs — this is your CI/CD evidence
for the CCA report.

---

## P. Git Commands

Basic commands you'll need throughout the project:

```powershell
# Initialize a repository (only once, if not already done)
git init

# Check status of changes
git status

# Stage changes
git add .

# Commit changes
git commit -m "Describe your change here"

# Connect to a GitHub remote (only once)
git remote add origin https://github.com/<your-username>/MLOps-CCA.git

# Push to GitHub
git push -u origin main

# Pull latest changes
git pull

# Create and switch to a new branch
git checkout -b feature/my-change

# Merge a branch back into main
git checkout main
git merge feature/my-change
```

---

## Q. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: config.yaml` | Running a script from the wrong folder | Run all commands from the `MLOps-CCA/` project root |
| `[MODEL LOADING ERROR] Could not find the trained model` | API started before training | Run `python src/train.py` first |
| `ModuleNotFoundError: No module named 'tensorflow'` (or similar) | Dependencies not installed / venv not activated | Activate venv, then `pip install -r requirements.txt` |
| PowerShell blocks `Activate.ps1` | Execution policy restriction | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once |
| `422 Unprocessable Entity` from `/predict` | Missing field, wrong type, or negative value sent | Check your JSON payload matches the 4 required fields, all positive numbers |
| MLflow UI shows no runs | `mlruns` folder not found / wrong working directory | Run `mlflow ui --backend-store-uri mlruns` from the project root, after training at least once |
| Evidently script fails with "datasets not found" | Monitoring script run before training | Run `python src/train.py` first (it exports the CSVs) |
| `python --version` shows something other than 3.10.x after activating the venv | The venv was created with the wrong Python (e.g. `python -m venv venv` picked up 3.13 instead of 3.10) | Delete the `venv` folder and recreate it explicitly with `py -3.10 -m venv venv` |
| `pytest tests/ -v` shows `ModuleNotFoundError: No module named 'api'` for all the API tests, while the model tests pass fine | Running the bare `pytest` command (unlike `python -m pytest`) doesn't add the project root to Python's import path, so `from api.app import app` can't find the `api` package | Already fixed: a `conftest.py` at the project root ensures the project root is always on the import path, no matter how pytest is invoked. If you still see this, make sure you're running the command from the `MLOps-CCA/` folder itself (`cd` into it first) |
| GitHub Actions workflow fails on install step | A pinned version in `requirements.txt` isn't available on the CI runner's Python version | Check the Actions log for the exact package/version and adjust if needed |

---

## R. How All the Tools Work Together

Think of this project as a relay race, where each tool hands off to the
next:

1. **scikit-learn** prepares clean, split, and scaled data.
2. **TensorFlow** learns patterns from that data and becomes a trained
   model.
3. **MLflow Tracking** records exactly *how* that model was trained
   (hyperparameters) and *how well* it performed (metrics), so nothing is
   lost between experiments.
4. **MLflow Model Registry** takes the best/latest trained model and gives
   it an official, versioned identity — separating "a model I trained" from
   "the model our system currently trusts".
5. **Git/GitHub** stores and versions all the *code* that produced that
   model, so the whole pipeline is reproducible by anyone.
6. **GitHub Actions** automatically re-validates the code (and re-trains +
   tests the model) every time it changes, catching breakages immediately.
7. **FastAPI** takes the registered model and exposes it to the outside
   world as a simple, documented REST API.
8. **Evidently AI** watches the data flowing through the system over time
   and raises a flag if it starts looking different from what the model
   was trained on — prompting a human to decide whether to retrain.

Together, these tools cover the full loop: **build → track → govern →
validate → serve → monitor → improve** — which is exactly what "MLOps"
means in practice.

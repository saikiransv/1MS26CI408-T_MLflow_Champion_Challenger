
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient


# Project configuration
ROOT = Path(__file__).resolve().parent

MODEL_NAME = "sentiment-model"
ALIAS = "champion"

INPUT_FILE = ROOT / "data" / "new_data.csv"
OUTPUT_FILE = ROOT / "outputs" / "new_data_predictions.csv"

# Connect to MLflow SQLite database
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

client = MlflowClient()


# Get Champion model version and metadata
print("\nLoading Champion from MLflow Registry...")

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    ALIAS
)

version = model_version.version
hf_model_id = model_version.tags.get(
    "hf_model_id",
    "Not available"
)

print("\nMODEL INFORMATION")
print("------------------------------")
print("Registered model:", MODEL_NAME)
print("Alias:", ALIAS)
print("Model version:", version)
print("Hugging Face model ID:", hf_model_id)


# Load model exclusively through MLflow alias
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

print("\nLoading model:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)


# Load completely new examples
new_data = pd.read_csv(INPUT_FILE)

if "text" not in new_data.columns:
    raise ValueError("Input CSV must contain a text column.")

print("\nNew examples:", len(new_data))

# Generate predictions
predictions = model.predict(
    new_data[["text"]]
)

# Combine input and predictions
results = pd.concat(
    [
        new_data.reset_index(drop=True),
        predictions.reset_index(drop=True)
    ],
    axis=1
)

# Add model metadata
results["model_version"] = version
results["hf_model_id"] = hf_model_id

# Save predictions
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)

print("\nPREDICTION RESULTS")
print("------------------------------")
print(results.to_string(index=False))

print("\nSaved predictions to:")
print(OUTPUT_FILE)

print("\nChampion prediction completed successfully!")

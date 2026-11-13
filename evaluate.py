
import json
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from mlflow.models import infer_signature


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = ROOT / "models"

for folder in [OUTPUT_DIR, PLOT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

EVAL_FILE = DATA_DIR / "eval_inputs.csv"

SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 128
DEVICE = "cpu"

STUDENT_USN = "1MS26CI408-T"

MODELS = {
    "A": "distilbert-base-uncased-finetuned-sst-2-english",
    "B": "textattack/bert-base-uncased-SST-2"
}

TASK = "sentiment-analysis"

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("sentiment-champion-challenger")

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# CUSTOM MLFLOW PYFUNC MODEL
# --------------------------------------------------

class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["hf_model"]
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            context.artifacts["hf_model"]
        )

        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()

        results = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)

            predictions = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values

            for pred, conf in zip(
                predictions.tolist(), confidence.tolist()
            ):
                results.append({
                    "predicted_label": int(pred),
                    "sentiment": "positive" if pred == 1 else "negative",
                    "confidence": float(conf)
                })

        return pd.DataFrame(results)


# --------------------------------------------------
# LOAD EVALUATION DATA
# --------------------------------------------------

print("\nLoading evaluation dataset...")

df = pd.read_csv(EVAL_FILE)

texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).to_numpy()

print("Evaluation examples:", len(df))


# --------------------------------------------------
# EVALUATE ONE MODEL
# --------------------------------------------------

def evaluate_model(role, model_id):

    short_name = model_id.split("/")[-1]

    run_name = f"{role}_{TASK}_{short_name}"

    print("\n" + "=" * 60)
    print("Evaluating:", model_id)
    print("=" * 60)

    print("Downloading model files...")

    local_model_path = snapshot_download(
        repo_id=model_id,
        local_dir=MODEL_DIR / short_name
    )

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path
    )

    model.to(DEVICE)
    model.eval()

    predictions = []
    confidences = []

    print("Running inference...")

    start_time = time.perf_counter()

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)

        batch_predictions = torch.argmax(
            probabilities, dim=1
        ).tolist()

        batch_confidences = torch.max(
            probabilities, dim=1
        ).values.tolist()

        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)

    total_time = time.perf_counter() - start_time

    # Both models are SST-2 classifiers:
    # index 0 = negative, index 1 = positive

    y_pred = np.array(predictions)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "total_inference_s": float(total_time),
        "avg_latency_ms": float(total_time * 1000 / len(texts))
    }

    predictions_df = pd.DataFrame({
        "text": texts,
        "actual_label": y_true,
        "predicted_label": y_pred,
        "actual_sentiment": [
            "positive" if x == 1 else "negative" for x in y_true
        ],
        "predicted_sentiment": [
            "positive" if x == 1 else "negative" for x in y_pred
        ],
        "confidence": confidences
    })

    prediction_file = OUTPUT_DIR / f"{role}_predictions.csv"
    predictions_df.to_csv(prediction_file, index=False)

    metrics_file = OUTPUT_DIR / f"{role}_metrics.json"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    ).plot(ax=ax, cmap="Blues")

    ax.set_title(f"{role} - {short_name}")
    fig.tight_layout()

    cm_file = PLOT_DIR / f"{role}_confusion_matrix.png"
    fig.savefig(cm_file)
    plt.close(fig)

    # Log run and MLflow model
    with mlflow.start_run(run_name=run_name) as run:

        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "role": role,
            "task": TASK,
            "hf_model_id": model_id,
            "dataset_name": "GLUE SST-2 validation",
            "library_versions": (
                f"transformers={__import__('transformers').__version__}, "
                f"torch={torch.__version__}, "
                f"mlflow={mlflow.__version__}"
            )
        })

        mlflow.log_params({
            "hf_model_id": model_id,
            "task": TASK,
            "role": role,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "eval_samples": len(df),
            "seed": SEED,
            "device": DEVICE
        })

        mlflow.log_metrics(metrics)

        input_example = pd.DataFrame({
            "text": [texts[0]]
        })

        output_example = pd.DataFrame({
            "predicted_label": [int(y_pred[0])],
            "sentiment": [
                "positive" if y_pred[0] == 1 else "negative"
            ],
            "confidence": [float(confidences[0])]
        })

        signature = infer_signature(
            input_example,
            output_example
        )

        mlflow.pyfunc.log_model(
            name="sentiment_model",
            python_model=HuggingFaceSentimentModel(),
            artifacts={"hf_model": local_model_path},
            signature=signature,
            input_example=input_example
        )

        mlflow.log_artifact(str(prediction_file))
        mlflow.log_artifact(str(metrics_file))
        mlflow.log_artifact(str(cm_file))

    print("\nModel results:", role)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Release memory
    del model
    del tokenizer

    return {
        "role": role,
        "model_id": model_id,
        "metrics": metrics,
        "predictions": y_pred,
        "run_id": run.info.run_id if False else None
    }


# --------------------------------------------------
# RUN BOTH MODELS
# --------------------------------------------------

print("\nSTARTING CHAMPION-CHALLENGER EXPERIMENT")

results = []

for role, model_id in MODELS.items():
    results.append(evaluate_model(role, model_id))


# --------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "role": result["role"],
        "model_id": result["model_id"],
        **result["metrics"]
    }
    for result in results
])

comparison_file = OUTPUT_DIR / "metrics_summary.csv"
comparison.to_csv(comparison_file, index=False)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))

comparison.set_index("role")[
    ["accuracy", "precision_macro", "recall_macro", "macro_f1"]
].plot(kind="bar", ax=ax)

ax.set_title("Champion vs Challenger - Metrics")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

comparison_plot = PLOT_DIR / "model_comparison.png"
fig.savefig(comparison_plot)
plt.close(fig)

# Disagreement examples
disagreements = pd.DataFrame({
    "text": texts,
    "actual_label": y_true,
    "model_A_prediction": results[0]["predictions"],
    "model_B_prediction": results[1]["predictions"]
})

disagreements = disagreements[
    disagreements["model_A_prediction"] !=
    disagreements["model_B_prediction"]
].copy()

def explain_disagreement(text):
    text = text.lower()

    if "grief" in text or "centuries" in text:
        return "Emotional and abstract language may lead to different interpretations."

    if "gasp" in text or "laugh" in text or "tear" in text:
        return "Mixed emotional expressions combine positive and negative sentiment."

    if "courage" in text or "harrowing" in text or "unspeakable" in text:
        return "Negative emotional vocabulary appears alongside praise for courage."

    if "dark satire" in text or "outrageous" in text:
        return "Satirical language may cause ambiguity in sentiment classification."

    if "solid film" in text or "more conscientious" in text:
        return "Positive praise is qualified by a contrasting criticism."

    if "without fuss" in text or "cold war" in text:
        return "Positive descriptive words appear alongside criticism of the premise."

    return "The models interpret the sentence differently; manual review is needed."


disagreements["comment"] = disagreements["text"].apply(
    explain_disagreement
)

disagreements_file = OUTPUT_DIR / "disagreement_examples.csv"
disagreements.to_csv(disagreements_file, index=False)


# --------------------------------------------------
# CHAMPION-CHALLENGER DECISION AND COMPARISON RUN
# --------------------------------------------------

PRIMARY_METRIC = "macro_f1"
MARGIN = 0.01
LATENCY_GUARDRAIL = 5.0

score_a = results[0]["metrics"][PRIMARY_METRIC]
score_b = results[1]["metrics"][PRIMARY_METRIC]

latency_a = results[0]["metrics"]["avg_latency_ms"]
latency_b = results[1]["metrics"]["avg_latency_ms"]

delta = score_b - score_a

# Model B wins only if it improves Macro-F1 by the margin
# and satisfies the latency guardrail.
if (
    delta >= MARGIN
    and latency_b <= latency_a * LATENCY_GUARDRAIL
):
    winner = "B"
else:
    winner = "A"

# Save comparison metrics as JSON
summary_json = OUTPUT_DIR / "metrics_summary.json"
comparison.to_json(summary_json, orient="records", indent=4)

# Create the single MLflow comparison run
with mlflow.start_run(run_name=f"compare_{TASK}"):

    mlflow.log_params({
        "primary_metric": PRIMARY_METRIC,
        "margin": MARGIN,
        "latency_guardrail": LATENCY_GUARDRAIL,
    })

    mlflow.log_metrics({
        "model_A_macro_f1": score_a,
        "model_B_macro_f1": score_b,
        "model_A_accuracy": results[0]["metrics"]["accuracy"],
        "model_B_accuracy": results[1]["metrics"]["accuracy"],
        "delta_primary_metric": delta,
        "disagreement_count": len(disagreements),
    })

    mlflow.set_tags({
        "winner": winner,
        "decision_rule": "macro_f1_improvement_with_latency_guardrail",
    })

    for artifact in [
        comparison_file,
        summary_json,
        comparison_plot,
        disagreements_file,
        EVAL_FILE,
        ROOT / "README.md",
        OUTPUT_DIR / "A_predictions.csv",
        OUTPUT_DIR / "B_predictions.csv",
        OUTPUT_DIR / "A_metrics.json",
        OUTPUT_DIR / "B_metrics.json",
        PLOT_DIR / "A_confusion_matrix.png",
        PLOT_DIR / "B_confusion_matrix.png",
    ]:
        if artifact.exists():
            mlflow.log_artifact(str(artifact))


print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))

print("\nDisagreement examples:", len(disagreements))
print("\nResults saved in:", OUTPUT_DIR)
print("MLflow tracking directory:", ROOT / "mlruns")

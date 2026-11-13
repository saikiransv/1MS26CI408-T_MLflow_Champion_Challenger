
from pathlib import Path
import time

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parent

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

client = MlflowClient()

EXPERIMENT_NAME = "sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

# Find the experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError("MLflow experiment not found.")

# Retrieve the completed A and B runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"]
)

model_runs = {}

for run in runs:
    role = run.data.tags.get("role")

    if role in ["A", "B"] and role not in model_runs:
        model_runs[role] = run

if "A" not in model_runs or "B" not in model_runs:
    raise RuntimeError(
        "Could not find both completed model runs in MLflow."
    )

# Register both models
versions = {}

for role in ["A", "B"]:

    run = model_runs[role]

    model_uri = f"runs:/{run.info.run_id}/sentiment_model"

    print(f"\nRegistering Model {role}")
    print("Run ID:", run.info.run_id)
    print("Model URI:", model_uri)

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    versions[role] = registered.version

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered.version,
        "hf_model_id",
        run.data.tags["hf_model_id"]
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered.version,
        "role",
        role
    )

    # Wait until registration is ready
    while True:
        current = client.get_model_version(
            REGISTERED_MODEL_NAME,
            registered.version
        )

        if current.status == "READY":
            break

        if current.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Registration failed for Model {role}"
            )

        time.sleep(2)

    print("Registered version:", registered.version)


# Read the latest comparison run's actual decision
comparison_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED' and tags.mlflow.runName = 'compare_sentiment-analysis'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if not comparison_runs:
    raise RuntimeError("No completed comparison run found.")

winner = comparison_runs[0].data.tags.get("winner")

if winner not in ["A", "B"]:
    raise RuntimeError("Comparison run has no valid winner tag.")

champion_role = winner
challenger_role = "A" if winner == "B" else "B"

print("Decision read from MLflow:", winner)


client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    versions[champion_role]
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    versions[challenger_role]
)

print("\n" + "=" * 55)
print("MODEL REGISTRY CONFIGURED")
print("=" * 55)

print("Registered model:", REGISTERED_MODEL_NAME)
print("Champion: Model B | Version", versions["B"])
print("Challenger: Model A | Version", versions["A"])
print("\nAliases assigned successfully.")

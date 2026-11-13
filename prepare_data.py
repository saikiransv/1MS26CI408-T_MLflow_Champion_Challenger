
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Reproducible evaluation dataset
SEED = 42
EVAL_SAMPLES = 100

print("Downloading GLUE SST-2 validation dataset...")

dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")

# Select the same fixed 100 examples every time
selected = dataset.shuffle(seed=SEED).select(range(EVAL_SAMPLES))

eval_df = pd.DataFrame({
    "text": selected["sentence"],
    "label": selected["label"]
})

eval_df.to_csv(DATA_DIR / "eval_inputs.csv", index=False)

# Eight original examples reserved for final Champion prediction
new_examples = [
    "The performance was outstanding and exceeded my expectations.",
    "I regret buying this product because it stopped working immediately.",
    "The story was interesting, but the ending felt disappointing.",
    "Absolutely loved the experience from beginning to end.",
    "The service was slow and the staff were unhelpful.",
    "It is an average experience with nothing particularly special.",
    "A brilliant film with excellent acting and beautiful visuals.",
    "I would never recommend this to anyone."
]

new_df = pd.DataFrame({
    "id": range(1, len(new_examples) + 1),
    "text": new_examples
})

new_df.to_csv(DATA_DIR / "new_data.csv", index=False)

print("\nDataset preparation completed!")
print(f"Evaluation examples: {len(eval_df)}")
print(f"New prediction examples: {len(new_df)}")
print("\nEvaluation label counts:")
print(eval_df["label"].value_counts().sort_index())
print("\nCreated files:")
print(DATA_DIR / "eval_inputs.csv")
print(DATA_DIR / "new_data.csv")

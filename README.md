# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This project implements a Champion–Challenger model comparison workflow using Hugging Face Transformers and MLflow.

Two pre-trained sentiment classification models are evaluated using the same evaluation dataset. Their performance is compared using classification metrics and inference latency. The selected Champion is registered in the MLflow Model Registry and used to generate predictions on new data.

## 2. Objectives

- Evaluate two Hugging Face sentiment classification models.
- Use identical evaluation inputs for a fair comparison.
- Calculate accuracy, macro precision, macro recall, and macro-F1.
- Measure inference latency.
- Track experiments and artifacts using MLflow.
- Select a Champion using a defined comparison rule.
- Register models and assign Champion and Challenger aliases.
- Load the registered Champion and predict new examples.

## 3. Models Used

| Model | Hugging Face Model ID |
|---|---|
| Model A – DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` |
| Model B – BERT | `textattack/bert-base-uncased-SST-2` |

Both models perform binary sentiment classification: Positive and Negative.

## 4. Dataset

The evaluation uses 100 examples from the GLUE SST-2 validation dataset.

Files:

- `data/eval_inputs.csv` – evaluation inputs and reference labels.
- `data/new_data.csv` – eight new examples for Champion prediction.

Both models are evaluated using the same 100 examples to maintain a consistent comparison.

## 5. Project Structure

```text
MLflow_Champion_Challenger/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── models/
│   ├── bert-base-uncased-SST-2/
│   └── distilbert-base-uncased-finetuned-sst-2-english/
│
├── outputs/
│   ├── A_metrics.json
│   ├── A_predictions.csv
│   ├── B_metrics.json
│   ├── B_predictions.csv
│   ├── disagreement_examples.csv
│   ├── metrics_summary.csv
│   ├── metrics_summary.json
│   └── new_data_predictions.csv
│
├── screenshots/
│   ├── 01_model_comparison.png
│   ├── 02_model_registry_aliases.png
│   └── 03_champion_prediction.png
│
├── evaluate.py
├── prepare_data.py
├── register_champion.py
├── predict_from_mlflow.py
├── test.py
├── requirements.txt
├── README.md
└── mlflow.db
```

## 6. Environment and Installation

Requirements:

- Python 3.12
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- Scikit-learn
- Matplotlib

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 7. Execution Steps

Run the following scripts from the project root directory.

### Step 1 – Prepare the dataset

```powershell
python prepare_data.py
```

### Step 2 – Evaluate and compare both models

```powershell
python evaluate.py
```

This step evaluates both models, calculates metrics, measures latency, records MLflow runs, and generates comparison artifacts.

### Step 3 – Register the Champion

```powershell
python register_champion.py
```

This registers the selected Champion and Challenger in the MLflow Model Registry and assigns their aliases.

### Step 4 – Predict new data

```powershell
python predict_from_mlflow.py
```

This loads the model through its Champion alias and generates predictions for the eight new examples.

### Step 5 – Open the MLflow interface

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the following address in a browser:

http://127.0.0.1:5000

## 8. Champion Selection Rule

The final comparison uses the following decision rule:

- Primary metric: Macro-F1.
- Direction: Higher Macro-F1 is better.
- Minimum improvement margin: 0.01.
- Latency guardrail: The candidate model's average latency must not exceed five times the other model's latency.

Model B is selected if it improves Macro-F1 by at least 0.01 and satisfies the latency guardrail. Otherwise, Model A remains Champion.

## 9. Final Experimental Results

Both models were evaluated on the same 100 examples.

| Metric | Model A – DistilBERT | Model B – BERT |
|---|---:|---:|
| Accuracy | 0.9100 | 0.9300 |
| Macro Precision | 0.909278 | 0.929803 |
| Macro Recall | 0.906773 | 0.927173 |
| Macro-F1 | 0.907928 | 0.928389 |
| Total Inference Time (s) | 1.799072 | 3.648160 |
| Average Latency (ms) | 17.990716 | 36.481595 |

### Comparison

- Macro-F1 improvement (B − A): 0.020461.
- Model B's average latency is approximately 2.03 times Model A's latency.
- Model B satisfies the five-times latency guardrail.
- Six prediction disagreements were identified between the models.

## 10. Final Champion Decision

Model B, TextAttack BERT, achieved a Macro-F1 score of 0.928389, compared with 0.907928 for Model A.

The improvement of 0.020461 exceeds the required margin of 0.01. Its latency also remains within the specified guardrail.

Therefore, Model B was selected as the Champion, while Model A was retained as the Challenger.

## 11. MLflow Model Registry

Registered model name: `sentiment-model`

| Model | Registered Version | Alias |
|---|---:|---|
| TextAttack BERT | 6 | champion |
| DistilBERT | 5 | challenger |

The Champion is loaded using the MLflow model URI:

```text
models:/sentiment-model@champion
```

## 12. New Data Prediction

The registered Champion was successfully loaded from the MLflow Model Registry.

- Number of new examples: 8
- Selected model: TextAttack BERT
- Registered version: 6
- Output file: `outputs/new_data_predictions.csv`

The prediction script saves the predicted sentiment and confidence information for each input example.

## 13. Screenshots and Evidence

The `screenshots/` directory contains:

1. `01_model_comparison.png` – MLflow experiment comparison.
2. `02_model_registry_aliases.png` – registered models and aliases.
3. `03_champion_prediction.png` – successful predictions using the Champion model.

## 14. Limitations

- Evaluation was performed on a fixed subset of 100 examples.
- Both models were evaluated using CPU inference.
- The small evaluation sample does not establish statistical significance.
- Latency can vary depending on system load and execution conditions.
- Some new-data predictions may require manual review; model confidence alone does not guarantee correctness.

## 15. Conclusion

This project demonstrates an end-to-end Champion–Challenger workflow using Hugging Face Transformers and MLflow. Two sentiment classification models were evaluated on the same 100 GLUE SST-2 validation examples. Their accuracy, macro precision, macro recall, macro-F1, and inference latency were measured and recorded.

Model B, TextAttack BERT, achieved a macro-F1 score of 0.928389, outperforming Model A, DistilBERT, which achieved 0.907928. Although Model B had higher inference latency, its average latency remained within the five-times guardrail. It therefore satisfied the comparison rule and was selected as the Champion.

The selected model was registered as version 6 of `sentiment-model`, while DistilBERT was retained as Challenger version 5. The Champion alias was then used to load the registered model and generate predictions for eight new examples.

Overall, this project demonstrates experiment tracking, model evaluation, performance comparison, model registration, alias-based deployment, and prediction on new data. It also highlights the importance of considering both predictive performance and inference cost when selecting a model for deployment.

## 16. AI Assistant Declaration

AI assistance was used for learning, implementation guidance, debugging, documentation preparation, and understanding the MLflow workflow. The project scripts were executed and the reported experimental results were obtained from the project environment.

## 17. Reproducibility Notes

- Run all commands from the project root directory.
- Use the same evaluation dataset for both models.
- The final comparison results are recorded in `outputs/metrics_summary.json`.
- The model predictions and disagreements are available in the `outputs/` directory.
- The registered Champion can be accessed through the `champion` alias in MLflow.
- The MLflow tracking database and downloaded model files are excluded from Git version control.
## 17. Reproducibility Notes

- Run all commands from the project root directory.
- Use the same evaluation dataset for both models.
- The final comparison results are recorded in `outputs/metrics_summary.json`.
- The model predictions and disagreements are available in the `outputs/` directory.
- The registered Champion can be accessed through the `champion` alias in MLflow.
- The MLflow tracking database and downloaded model files are excluded from Git version control.

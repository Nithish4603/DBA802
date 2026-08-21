# DBA802 - AI-Driven Cybersecurity Strategic Decision Intelligence

## Research Topic

**AI-Driven Predictive Analytics for Cybersecurity Threat Detection and Strategic Decision Intelligence**

This repository contains a complete local experimental pipeline using a prepared UNSW-NB15 subset.

## Pipeline

Dataset -> preprocessing -> Logistic Regression / Random Forest / XGBoost / MLP -> evaluation -> explainability -> risk scoring -> strategic recommendations.

## Dataset Included

- `data/prepared/train_local.csv`: 50,000 rows
- `data/prepared/test_local.csv`: 20,000 rows
- `label`: 0 = Normal, 1 = Attack
- `attack_cat`: original attack category, retained for interpretation but excluded from binary model features

## 1. Clone

```bash
git clone https://github.com/Nithish4603/DBA802.git
cd DBA802
```

If you are using the project branch:

```bash
git checkout cybersecurity-ai
```

## 2. Create environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Fast test first

```bash
python run_experiment.py --quick
```

This uses about 5,000 training rows and 2,000 test rows.

## 5. Full local experiment

```bash
python run_experiment.py
```

This runs all four models on the 50k/20k prepared dataset.

## Models

1. Logistic Regression
2. Random Forest
3. XGBoost
4. MLP Neural Network

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training time
- Inference time

## Outputs

`outputs/metrics/model_comparison.csv` contains the main research table.

`outputs/figures/` contains confusion matrices, ROC curves, model comparison, and feature importance.

`outputs/predictions/strategic_decision_output.csv` contains attack probability, risk level, and recommended strategic action.

## Research Rule

Do not write final numerical claims in the paper until the full experiment has actually been run and the saved result files have been inspected.

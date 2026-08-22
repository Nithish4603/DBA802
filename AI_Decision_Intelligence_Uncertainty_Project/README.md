# AI-Driven Predictive and Prescriptive Analytics for Intelligent Decision Support Under Uncertainty

## Research implementation

This repository is a local, CPU-friendly experimental implementation for the topic:

**AI-Driven Predictive and Prescriptive Analytics for Intelligent Decision Support Under Uncertainty**

The application domain is **building energy-efficiency decision support**. The experiment predicts heating and cooling loads, quantifies predictive uncertainty with split conformal prediction, and then searches feasible design alternatives to recommend lower-load configurations.

## Dataset

The project includes `data/energy_efficiency.csv`, a CSV representation of the UCI Energy Efficiency dataset by Tsanas and Xifara. The official UCI dataset contains 768 simulated building configurations, eight input features, and two continuous targets: Heating Load and Cooling Load.

Official source: https://archive.ics.uci.edu/dataset/242/energy+efficiency
Dataset DOI: https://doi.org/10.24432/C51307
License: CC BY 4.0

### Features

- X1 / `relative_compactness`
- X2 / `surface_area`
- X3 / `wall_area`
- X4 / `roof_area`
- X5 / `overall_height`
- X6 / `orientation`
- X7 / `glazing_area`
- X8 / `glazing_distribution`

Targets:
- Y1 / `heating_load`
- Y2 / `cooling_load`

## What the code does

1. Loads and validates the dataset.
2. Creates reproducible train/calibration/test partitions (60/20/20).
3. Trains multiple regression models:
   - Linear Regression
   - Random Forest
   - Gradient Boosting
   - MLP neural network
   - XGBoost (when installed)
4. Compares MAE, RMSE, R², training time, and inference time.
5. Selects the model with the lowest mean RMSE.
6. Builds 90% split-conformal prediction intervals for heating and cooling load.
7. Creates data-driven Low/Moderate/High/Critical decision tiers from predicted total energy load.
8. Runs a prescriptive search over feasible orientation/glazing settings while holding structural geometry fixed.
9. Generates publication-ready CSV outputs and plots.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run_experiment.py
```

## Key output files

`outputs/metrics/model_comparison.csv`  
Model-level comparison.

`outputs/metrics/model_target_metrics.csv`  
Heating/cooling metrics for every model.

`outputs/metrics/conformal_uncertainty.csv`  
Nominal and empirical interval coverage.

`outputs/metrics/feature_importance.csv`  
Permutation importance for the selected model.

`outputs/predictions/test_predictions_with_uncertainty.csv`  
Actual values, predictions, 90% uncertainty bounds, and decision tiers.

`outputs/predictions/prescriptive_recommendations.csv`  
Current vs recommended design settings and estimated load reduction.

## Generated figures

- `model_comparison_r2.png`
- `model_comparison_rmse.png`
- `actual_vs_predicted_heating_load.png`
- `actual_vs_predicted_cooling_load.png`
- `uncertainty_intervals_heating.png`
- `feature_importance.png`
- `decision_tier_distribution.png`
- `prescriptive_improvement.png`

## Important research interpretation

The prescriptive layer is a constrained **data-driven alternative search**, not a claim of globally optimal building design. It varies only orientation, glazing area, and glazing distribution over values observed in the dataset while holding structural geometry fixed. The conformal intervals quantify marginal predictive uncertainty under the held-out split; they are not guarantees under arbitrary distribution shift.

## What to send back for paper preparation

After running the project, send these files:

- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/model_target_metrics.csv`
- `outputs/metrics/conformal_uncertainty.csv`
- `outputs/metrics/feature_importance.csv`
- `outputs/metrics/prescriptive_summary.csv`
- `outputs/predictions/prescriptive_recommendations.csv`
- all PNG files from `outputs/figures/`

Those results can then be used to write the Results, Discussion, uncertainty-analysis, and prescriptive-decision sections of the paper.

# Research Design

## Topic
AI-Driven Predictive and Prescriptive Analytics for Intelligent Decision Support Under Uncertainty

## Research question
How can predictive AI, uncertainty quantification, and constrained prescriptive analytics be integrated to provide reliable and actionable decision support for building energy-efficiency design?

## Experimental logic

Data -> predictive models -> model comparison -> uncertainty quantification -> explainability -> decision tiers -> prescriptive alternative search -> recommended action.

## Predictive component
Heating Load and Cooling Load are treated as continuous regression targets. Multiple models are evaluated under a common held-out test set.

## Uncertainty component
Split conformal prediction is applied using a dedicated calibration set. Separate 90% marginal intervals are produced for heating and cooling load.

## Prescriptive component
For each held-out design, the selected predictive model evaluates alternative combinations of orientation, glazing area, and glazing distribution observed in the dataset. Structural geometry is held fixed. The recommendation minimizes predicted combined heating/cooling load with a mild penalty for large glazing-area changes.

## Evaluation
Predictive metrics: MAE, RMSE, R².  
Uncertainty metrics: empirical coverage and interval width.  
Prescriptive metrics: estimated absolute and percentage reduction in predicted total load.

## Threats to validity
- Small benchmark dataset (768 simulated configurations).
- Simulation-based rather than live building measurements.
- Prescriptions are restricted to observed discrete facade/orientation choices.
- Predictive improvement does not imply real-world causal savings.
- Conformal coverage may degrade under distribution shift.

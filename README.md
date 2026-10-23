# K-AIDE proof-of-concept

This compact repository accompanies the short communication **Data-Driven Knowledge-Based AI for Digital Transformation, Innovation, and Entrepreneurship: An Explainable Opportunity-Assessment Framework** by **Nithish Kumar Maley**, Doctorate of Business Administration (DBA) Program, Midwest University, 851 Parr Rd, Wentzville, MO 63385, USA.

The code is intentionally small: one executable Python script generates a deterministic illustrative benchmark, evaluates three estimators across five fixed seeds, performs a 2 x 2 component ablation, and creates the two manuscript figures plus optional diagnostic outputs. The benchmark is synthetic and is included for reproducibility. It must not be presented as evidence about a particular industry or population without replacing it with approved real data.

## Run

```bash
python run_experiment.py --n-samples 3000 --seeds 42 43 44 45 46 --alpha 0.65 --output figures
```

The script prints repeated-seed mean and standard deviation for ROC-AUC, PR-AUC, Brier score, and accuracy; the ablation metrics; and paired squared-error tests. It writes:

- `figures/figure_1_correlation_heatmap.png`
- `figures/figure_2_roc_pr_curves.png`
- `figures/diagnostic_feature_importance.png` (optional)
- `figures/diagnostic_confusion_matrices.png` (optional)
- `figures/diagnostic_calibration_curve.png` (optional)
- `results/metrics_summary.csv`
- `results/ablation_summary.csv`
- `results/blend_sensitivity.csv`
- `results/heldout_predictions.csv`
- `results/paired_tests.csv`

The combined synthetic dataset is supplied as `synthetic_benchmark.csv`. It contains 15,000 records across seeds 42, 43, 44, 45, and 46, with a `seed`, `record_id`, ten normalized indicators, and the simulated `adopt` outcome. To run directly from the supplied CSV, use:

```bash
python run_experiment.py --dataset-file synthetic_benchmark.csv --n-samples 3000 --seeds 42 43 44 45 46 --alpha 0.65 --output figures --results-dir results
```

The generated figure files are high-resolution PNGs suitable for single-column placement. Figures 1 and 2 are the manuscript figures: the supplied feature-correlation audit and pooled ROC/PR curves. The remaining diagnostic outputs are optional supplements and should not be included in the main submission when a two-figure limit applies. The manuscript reports the exact thresholds, model settings, seeds, split, and blend weight used to produce the published tables and figures.

## Publication metadata

- Corresponding author: Nithish Kumar Maley
- Corresponding-author email: nithish.maley4603@gmail.com
- Repository: https://github.com/Nithish4603/DBA802
- Data availability: the supplied benchmark is synthetic and deterministic; the CSV and generator are included for auditability.

## Reproducibility notes

- Python 3.10 or newer is recommended.
- The five fixed seeds reproduce the benchmark and figures used in the manuscript.
- The supplied CSV and the generator produce the same benchmark; the CSV is included for auditability and convenience.
- The knowledge rules are defined before the train/test split and are not learned from test labels. The weighted-criteria coefficients in `knowledge_score_and_trace` mirror the linear value/feasibility/risk terms used by `make_benchmark`, so the knowledge-only estimator is a close (but non-interaction) oracle rather than an arbitrary rule set.
- The output includes a rule trace for every held-out opportunity and a blend-sensitivity sweep.
- The alpha-sensitivity sweep and the optional feature-importance, confusion-matrix, and calibration diagnostics are computed from the same fixed-seed benchmark for auditability; only the two named manuscript figures are required for the short communication.
- The published run uses single-threaded random-forest execution (`n_jobs=1`) to avoid parallel aggregation drift. Install the exact versions pinned in `requirements.txt` before reproducing the published artifacts.
- For an empirical submission, replace `make_benchmark` with an approved dataset, document feature provenance, tune the blend parameter on validation data, and deposit the code/data release in a public repository.

For the final citation, quartile, and submission-readiness audit, see `reference_verification_report.md`. Quartile status is reported with a database/category/year caveat because no single ranking system assigns a universal Q1 label.

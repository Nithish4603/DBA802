# K-AIDE proof-of-concept

This compact repository accompanies the short communication **Data-Driven Knowledge-Based AI for Digital Transformation, Innovation, and Entrepreneurship: An Explainable Opportunity-Assessment Framework**.

The code is intentionally small: one executable Python script generates a deterministic illustrative benchmark, evaluates three estimators across five fixed seeds, performs a 2 x 2 component ablation, and creates the two manuscript figures plus five optional diagnostic figures. The benchmark is synthetic and is included for reproducibility. It must not be presented as evidence about a particular industry or population without replacing it with approved real data.

## Run

```bash
python run_experiment.py --n-samples 3000 --seeds 42 43 44 45 46 --alpha 0.65 --output figures
```

The script prints repeated-seed mean and standard deviation for ROC-AUC, PR-AUC, Brier score, and accuracy; the ablation metrics; and paired squared-error tests. It writes:

- `figures/figure_1_framework.png`
- `figures/figure_2_results.png`
- `figures/figure_3_correlation_heatmap.png`
- `figures/figure_4_feature_importance.png`
- `figures/figure_5_roc_pr_curves.png`
- `figures/figure_6_confusion_matrices.png`
- `figures/figure_7_calibration_curve.png`
- `results/metrics_summary.csv`
- `results/ablation_summary.csv`
- `results/blend_sensitivity.csv`
- `results/heldout_predictions.csv`
- `results/paired_tests.csv`

The combined synthetic dataset is supplied as `synthetic_benchmark.csv`. It contains 15,000 records across seeds 42, 43, 44, 45, and 46, with a `seed`, `record_id`, ten normalized indicators, and the simulated `adopt` outcome. To run directly from the supplied CSV, use:

```bash
python run_experiment.py --dataset-file synthetic_benchmark.csv --n-samples 3000 --seeds 42 43 44 45 46 --alpha 0.65 --output figures --results-dir results
```

The figure files are generated as 1000 dpi PNGs suitable for single-column placement. Figures 1 and 2 are the manuscript figures; Figures 3-7 are optional diagnostic supplements and should not be included in the main submission when a two-figure limit applies. The manuscript reports the exact thresholds, model settings, seeds, split, and blend weight used to produce the published tables and figures.

## Reproducibility notes

- Python 3.10 or newer is recommended.
- The five fixed seeds reproduce the benchmark and figures used in the manuscript.
- The supplied CSV and the generator produce the same benchmark; the CSV is included for auditability and convenience.
- The knowledge rules are defined before the train/test split and are not learned from test labels. The weighted-criteria coefficients in `knowledge_score_and_trace` mirror the linear value/feasibility/risk terms used by `make_benchmark`, so the knowledge-only estimator is a close (but non-interaction) oracle rather than an arbitrary rule set.
- The output includes a rule trace for every held-out opportunity and a blend-sensitivity sweep.
- Figures 3-7 are diagnostic supplements (correlation heatmap, feature importance, ROC/PR curves, confusion matrices, calibration curve) computed by pooling the held-out predictions across all five seeds.
- The published run uses single-threaded random-forest execution (`n_jobs=1`) to avoid parallel aggregation drift. Install the exact versions pinned in `requirements.txt` before reproducing the published artifacts. Even with pinned versions and `n_jobs=1`, a clean rerun has been observed to differ from the published metrics by up to about 0.0005 in ROC-AUC/Brier/accuracy and by up to roughly 30% in the (already very small) Wilcoxon p-values, most likely from BLAS/platform-level floating-point non-associativity rather than from the random-number generator. Treat the published numbers as representative at the reported precision rather than as bit-exact across machines.
- For an empirical submission, replace `make_benchmark` with an approved dataset, document feature provenance, tune the blend parameter on validation data, and deposit the code/data release in a public repository.

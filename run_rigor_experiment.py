"""
Adds the methodological rigor a journal-level paper needs on top of run_experiment.py:
stratified k-fold cross-validation, paired significance testing between the top two
models, and randomized hyperparameter search for XGBoost and Random Forest.

Does not modify run_experiment.py or its outputs; writes to outputs/metrics/ under
separate filenames so the original single-split results stay intact for comparison.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

from src.data import load_data
from src.models import build_models

ROOT = Path(__file__).resolve().parent
SEED = 42


def run_cross_validation(X_train, y_train, n_splits=5):
    models = build_models(X_train, seed=SEED)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fold_records = []
    summary_records = []
    for name, pipeline in models.items():
        print(f"\n{n_splits}-fold CV: {name}")
        start = time.perf_counter()
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.perf_counter() - start
        for fold_idx in range(n_splits):
            fold_records.append({
                "model": name,
                "fold": fold_idx + 1,
                "accuracy": result["test_accuracy"][fold_idx],
                "precision": result["test_precision"][fold_idx],
                "recall": result["test_recall"][fold_idx],
                "f1_score": result["test_f1"][fold_idx],
                "roc_auc": result["test_roc_auc"][fold_idx],
            })
        summary_records.append({
            "model": name,
            "f1_mean": result["test_f1"].mean(),
            "f1_std": result["test_f1"].std(),
            "accuracy_mean": result["test_accuracy"].mean(),
            "accuracy_std": result["test_accuracy"].std(),
            "roc_auc_mean": result["test_roc_auc"].mean(),
            "roc_auc_std": result["test_roc_auc"].std(),
            "cv_seconds": elapsed,
        })
        print(f"  F1 = {result['test_f1'].mean():.4f} +/- {result['test_f1'].std():.4f}")

    return pd.DataFrame(fold_records), pd.DataFrame(summary_records).sort_values("f1_mean", ascending=False)


def run_significance_test(fold_df, model_a, model_b):
    a = fold_df[fold_df["model"] == model_a].sort_values("fold")["f1_score"].to_numpy()
    b = fold_df[fold_df["model"] == model_b].sort_values("fold")["f1_score"].to_numpy()
    t_stat, t_p = stats.ttest_rel(a, b)
    w_stat, w_p = stats.wilcoxon(a, b)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_a_f1_mean": float(a.mean()),
        "model_b_f1_mean": float(b.mean()),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "significant_at_0.05": bool(t_p < 0.05),
    }


def run_hyperparameter_search(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from src.models import build_preprocessor
    from sklearn.pipeline import Pipeline

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {}

    print("\nRandomized search: xgboost")
    xgb_pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED)),
    ])
    xgb_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_dist, n_iter=15, cv=cv, scoring="f1", random_state=SEED, n_jobs=1, verbose=1
    )
    start = time.perf_counter()
    xgb_search.fit(X_train, y_train)
    results["xgboost"] = {
        "best_f1_cv": xgb_search.best_score_,
        "best_params": {k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()},
        "search_seconds": time.perf_counter() - start,
        "baseline_f1_cv": None,
    }

    print("\nRandomized search: random_forest")
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    ])
    rf_dist = {
        "model__n_estimators": [150, 200, 300, 400],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_dist, n_iter=15, cv=cv, scoring="f1", random_state=SEED, n_jobs=1, verbose=1
    )
    start = time.perf_counter()
    rf_search.fit(X_train, y_train)
    results["random_forest"] = {
        "best_f1_cv": rf_search.best_score_,
        "best_params": {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()},
        "search_seconds": time.perf_counter() - start,
        "baseline_f1_cv": None,
    }

    return results


def main():
    data_dir = ROOT / "data" / "prepared"
    outputs = ROOT / "outputs" / "metrics"
    outputs.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, meta_test = load_data(data_dir, quick=False)
    print(f"Training rows: {len(X_train):,} | Features: {X_train.shape[1]}")

    fold_df, summary_df = run_cross_validation(X_train, y_train, n_splits=5)
    fold_df.to_csv(outputs / "cv_fold_results.csv", index=False)
    summary_df.to_csv(outputs / "cv_summary.csv", index=False)
    print("\nCV summary (sorted by mean F1):")
    print(summary_df.to_string(index=False))

    top_two = summary_df["model"].tolist()[:2]
    sig = run_significance_test(fold_df, top_two[0], top_two[1])
    (outputs / "significance_test.json").write_text(json.dumps(sig, indent=2))
    print(f"\nSignificance test ({top_two[0]} vs {top_two[1]}):")
    print(json.dumps(sig, indent=2))

    tuning = run_hyperparameter_search(X_train, y_train)
    for name in ["xgboost", "random_forest"]:
        baseline_row = summary_df[summary_df["model"] == name]
        if not baseline_row.empty:
            tuning[name]["baseline_f1_cv"] = float(baseline_row["f1_mean"].iloc[0])
    (outputs / "hyperparameter_search.json").write_text(json.dumps(tuning, indent=2))
    print("\nHyperparameter search results:")
    print(json.dumps(tuning, indent=2))

    print("\nDone. See outputs/metrics/cv_fold_results.csv, cv_summary.csv, significance_test.json, hyperparameter_search.json")


if __name__ == "__main__":
    main()

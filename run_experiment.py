from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from src.data import load_data
from src.models import build_models
from src.evaluation import evaluate_model, plot_model_comparison
from src.decision_intelligence import build_decision_output
from src.explainability import create_permutation_importance

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 cybersecurity decision-intelligence experiment")
    parser.add_argument("--quick", action="store_true", help="Run on a small sample for a fast local smoke test")
    args = parser.parse_args()

    data_dir = ROOT / "data" / "prepared"
    outputs = ROOT / "outputs"
    for sub in ["models", "metrics", "figures", "predictions"]:
        (outputs / sub).mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, meta_test = load_data(data_dir, quick=args.quick)
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,} | Features: {X_train.shape[1]}")

    models = build_models(X_train)
    results = []
    model_outputs = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, pred, proba, report = evaluate_model(name, model, X_train, y_train, X_test, y_test, outputs / "figures")
        results.append(metrics)
        report.to_csv(outputs / "metrics" / f"classification_report_{name}.csv")
        joblib.dump(model, outputs / "models" / f"{name}.joblib")
        model_outputs[name] = (pred, proba, model)
        print(pd.Series(metrics).to_string())

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False).reset_index(drop=True)
    results_df.to_csv(outputs / "metrics" / "model_comparison.csv", index=False)
    plot_model_comparison(results_df, outputs / "figures")

    best_name = results_df.iloc[0]["model"]
    pred, proba, best_model = model_outputs[best_name]
    decisions = build_decision_output(meta_test, pred, proba)
    decisions.to_csv(outputs / "predictions" / "strategic_decision_output.csv", index=False)

    print(f"\nBest model by F1-score: {best_name}")
    print("Creating model-agnostic feature importance...")
    create_permutation_importance(best_model, X_test, y_test, outputs / "figures")

    summary = {
        "best_model": best_name,
        "quick_mode": args.quick,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    (outputs / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nExperiment complete. See outputs/metrics, outputs/figures, outputs/models, and outputs/predictions.")

if __name__ == "__main__":
    main()

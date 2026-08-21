from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report
)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)
    inference_seconds = time.perf_counter() - start
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "training_seconds": train_seconds,
        "inference_seconds": inference_seconds,
    }
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred), display_labels=["Normal", "Attack"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"confusion_matrix_{name}.png", dpi=200)
    plt.close()
    RocCurveDisplay.from_predictions(y_test, proba, name=name)
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"roc_curve_{name}.png", dpi=200)
    plt.close()
    report = classification_report(y_test, pred, target_names=["Normal", "Attack"], output_dict=True, zero_division=0)
    return metrics, pred, proba, pd.DataFrame(report).T


def plot_model_comparison(results_df, fig_dir):
    fig_dir = Path(fig_dir)
    cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ax = results_df.set_index("model")[cols].plot(kind="bar", figsize=(10, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_performance_comparison.png", dpi=200)
    plt.close()

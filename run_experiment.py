"""Reproducible proof-of-concept for the K-AIDE short communication.

The script generates a controlled illustrative benchmark, compares a data-only
AI model with an explicit knowledge layer and their hybrid, performs a small
component ablation, and writes the two manuscript figures plus three optional
diagnostic figures. The benchmark is synthetic by design; replace
``make_benchmark`` with approved real data before making sector-level claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "market_signal",
    "customer_need",
    "data_readiness",
    "digital_maturity",
    "innovation_novelty",
    "ecosystem_support",
    "resource_capacity",
    "implementation_complexity",
    "regulatory_exposure",
    "organizational_agility",
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
FIGURE_DPI = 1000


def make_benchmark(n: int, seed: int) -> pd.DataFrame:
    """Create a controlled opportunity-assessment benchmark in [0, 1]."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.beta(2.2, 2.2, size=(n, len(FEATURES))), columns=FEATURES)

    # Nonlinear value, feasibility, risk, and market-need interaction.
    value = (
        0.22 * x.market_signal
        + 0.15 * x.customer_need
        + 0.16 * x.innovation_novelty
        + 0.11 * x.ecosystem_support
        + 0.08 * x.organizational_agility
    )
    feasibility = (
        0.16 * x.data_readiness
        + 0.12 * x.digital_maturity
        + 0.12 * x.resource_capacity
        - 0.15 * x.implementation_complexity
    )
    risk = 0.12 * x.regulatory_exposure
    interaction = 0.18 * x.market_signal * x.customer_need
    latent = value + feasibility + interaction - risk

    # The benchmark contains explicit knowledge rules that are fixed before
    # the train/test split and are not learned from test labels.
    hard_gate = (
        (x.data_readiness < 0.22) & (x.digital_maturity < 0.25)
    ) | ((x.regulatory_exposure > 0.82) & (x.data_readiness < 0.45))
    synergy = (x.market_signal > 0.62) & (x.innovation_novelty > 0.68)
    latent = latent + 0.10 * synergy - 0.32 * hard_gate
    probability = 1.0 / (1.0 + np.exp(-8.0 * (latent - 0.48)))
    x["adopt"] = rng.binomial(1, probability)
    return x


def build_dataset(n_samples: int, seeds: list[int]) -> pd.DataFrame:
    """Build one combined, reproducible dataset with seed and record metadata."""
    frames = []
    for seed in seeds:
        frame = make_benchmark(n_samples, seed).copy()
        frame.insert(0, "seed", seed)
        frame.insert(1, "record_id", np.arange(n_samples, dtype=int))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_dataset(path: Path, seeds: list[int], n_samples: int) -> dict[int, pd.DataFrame]:
    """Load a combined dataset CSV and return one frame per requested seed."""
    frame = pd.read_csv(path)
    required = {"seed", "record_id", *FEATURES, "adopt"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not bool(frame["adopt"].isin([0, 1]).all()):
        raise ValueError("The adopt column must contain only 0/1 values.")
    by_seed = {}
    for seed in seeds:
        subset = frame.loc[frame["seed"] == seed].sort_values("record_id").copy()
        if len(subset) < n_samples:
            raise ValueError(f"Dataset has {len(subset)} records for seed {seed}; {n_samples} required.")
        by_seed[seed] = subset.head(n_samples).reset_index(drop=True)
    return by_seed


def knowledge_score_and_trace(frame: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Return a normalized knowledge score and the active rule trace."""
    base = (
        0.22 * frame.market_signal
        + 0.15 * frame.customer_need
        + 0.16 * frame.data_readiness
        + 0.12 * frame.digital_maturity
        + 0.16 * frame.innovation_novelty
        + 0.11 * frame.ecosystem_support
        + 0.08 * frame.organizational_agility
        + 0.12 * frame.resource_capacity
        - 0.15 * frame.implementation_complexity
        - 0.12 * frame.regulatory_exposure
    )
    low_readiness_gate = (frame.data_readiness < 0.22) & (frame.digital_maturity < 0.25)
    regulatory_gate = (frame.regulatory_exposure > 0.82) & (frame.data_readiness < 0.45)
    hard_gate = low_readiness_gate | regulatory_gate
    synergy = (frame.market_signal > 0.62) & (frame.innovation_novelty > 0.68)
    score = base + 0.10 * synergy - 0.32 * hard_gate

    traces: list[list[str]] = []
    for i in range(len(frame)):
        active: list[str] = []
        if low_readiness_gate.iloc[i]:
            active.append("low_readiness_gate")
        if regulatory_gate.iloc[i]:
            active.append("regulatory_gate")
        if synergy.iloc[i]:
            active.append("market_novelty_synergy")
        traces.append(active)
    return np.clip((score + 0.20) / 1.10, 0.0, 1.0).to_numpy(), traces


def evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return ranking, calibration, and classification metrics."""
    labels = (probability >= 0.5).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
        "Brier score": brier_score_loss(y_true, probability),
        "Accuracy": accuracy_score(y_true, labels),
    }


def run_once(n_samples: int, seed: int, alpha: float, source: pd.DataFrame | None = None) -> dict:
    """Fit all estimators for one seed and return test-level artifacts."""
    data = source.copy() if source is not None else make_benchmark(n_samples, seed)
    metadata = data[["seed", "record_id"]].copy() if {"seed", "record_id"}.issubset(data.columns) else None
    model_frame = data[FEATURES + ["adopt"]].copy()
    train, test = train_test_split(
        model_frame, test_size=0.30, random_state=seed, stratify=model_frame.adopt
    )
    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        # Single-threaded execution avoids small floating-point changes from
        # parallel tree aggregation and makes the published artifacts easier
        # to reproduce across machines.
        n_jobs=1,
    )
    model.fit(train[FEATURES], train.adopt)
    p_data = model.predict_proba(test[FEATURES])[:, 1]
    p_kb, trace = knowledge_score_and_trace(test[FEATURES])
    p_hybrid = np.clip(alpha * p_data + (1.0 - alpha) * p_kb, 0.0, 1.0)
    y = test.adopt.to_numpy()
    metrics = {
        "Data-only AI": evaluate(y, p_data),
        "Knowledge rules": evaluate(y, p_kb),
        "Hybrid K-AIDE": evaluate(y, p_hybrid),
    }
    return {
        "metrics": metrics,
        "y": y,
        "p_data": p_data,
        "p_kb": p_kb,
        "p_hybrid": p_hybrid,
        "trace_rate": float(np.mean([bool(item) for item in trace])),
        "test_index": test.index.to_numpy(),
        "test_metadata": metadata.loc[test.index].reset_index(drop=True) if metadata is not None else None,
        "model": model,
    }


def summarize(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metric means and standard deviations across repeated seeds."""
    records = []
    for run in runs:
        for method, metrics in run["metrics"].items():
            records.append({"method": method, **metrics})
    frame = pd.DataFrame(records).set_index("method")
    return frame.groupby(level=0).mean(), frame.groupby(level=0).std(ddof=1)


def alpha_sweep(runs: list[dict], alphas: np.ndarray) -> pd.DataFrame:
    """Measure blend sensitivity using the same held-out records."""
    rows = []
    for alpha in alphas:
        values = []
        for run in runs:
            p = np.clip(alpha * run["p_data"] + (1.0 - alpha) * run["p_kb"], 0.0, 1.0)
            values.append(evaluate(run["y"], p))
        rows.append({"alpha": alpha, **pd.DataFrame(values).mean().to_dict()})
    return pd.DataFrame(rows)


def paired_p_value(runs: list[dict], left: str, right: str) -> float:
    """Wilcoxon test on paired per-record squared errors across seeds."""
    left_losses, right_losses = [], []
    for run in runs:
        y = run["y"]
        left_p = run["p_hybrid"] if left == "Hybrid K-AIDE" else run["p_data"] if left == "Data-only AI" else run["p_kb"]
        right_p = run["p_hybrid"] if right == "Hybrid K-AIDE" else run["p_data"] if right == "Data-only AI" else run["p_kb"]
        left_losses.extend((left_p - y) ** 2)
        right_losses.extend((right_p - y) ** 2)
    result = wilcoxon(left_losses, right_losses, alternative="two-sided", zero_method="wilcox")
    return float(result.pvalue)


def ablation_summary(runs: list[dict]) -> pd.DataFrame:
    """Evaluate the 2 x 2 presence/absence of the data and knowledge layers."""
    records = []
    for name in ["Neither", "Data-only AI", "Knowledge rules", "Hybrid K-AIDE"]:
        values = []
        for run in runs:
            if name == "Neither":
                p = np.full(len(run["y"]), 0.5)
            elif name == "Data-only AI":
                p = run["p_data"]
            elif name == "Knowledge rules":
                p = run["p_kb"]
            else:
                p = run["p_hybrid"]
            values.append(evaluate(run["y"], p))
        row = pd.DataFrame(values).mean()
        row["sd Brier"] = pd.DataFrame(values)["Brier score"].std(ddof=1)
        row["sd Accuracy"] = pd.DataFrame(values)["Accuracy"].std(ddof=1)
        records.append({"Variant": name, **row.to_dict()})
    return pd.DataFrame(records).set_index("Variant")


def predictions_frame(runs: list[dict], seeds: list[int]) -> pd.DataFrame:
    """Return held-out predictions for auditing and manuscript-table regeneration."""
    frames = []
    for run, seed in zip(runs, seeds):
        frame = pd.DataFrame({
            "seed": seed,
            "y_true": run["y"],
            "p_data": run["p_data"],
            "p_knowledge": run["p_kb"],
            "p_hybrid": run["p_hybrid"],
        })
        if run["test_metadata"] is not None:
            frame.insert(1, "record_id", run["test_metadata"]["record_id"].to_numpy())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_results(
    results_dir: Path,
    means: pd.DataFrame,
    stds: pd.DataFrame,
    ablation: pd.DataFrame,
    sweep: pd.DataFrame,
    runs: list[dict],
    seeds: list[int],
    p_data: float,
    p_rules: float,
) -> None:
    """Write compact CSV artifacts used to audit and update the manuscript."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = means.add_suffix("_mean").join(stds.add_suffix("_sd"))
    summary.to_csv(results_dir / "metrics_summary.csv")
    ablation.to_csv(results_dir / "ablation_summary.csv")
    sweep.to_csv(results_dir / "blend_sensitivity.csv", index=False)
    predictions_frame(runs, seeds).to_csv(results_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame([
        {"comparison": "Hybrid K-AIDE vs Data-only AI", "test": "Wilcoxon paired squared error", "p_value": p_data},
        {"comparison": "Hybrid K-AIDE vs Knowledge rules", "test": "Wilcoxon paired squared error", "p_value": p_rules},
    ]).to_csv(results_dir / "paired_tests.csv", index=False)


def make_framework_figure(path: Path) -> None:
    """Write the conceptual K-AIDE information-flow figure."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=FIGURE_DPI)
    ax.axis("off")
    boxes = [
        (0.02, 0.57, 0.24, 0.28, "Business data\nmarket; customer\nreadiness; risk"),
        (0.02, 0.15, 0.24, 0.30, "Knowledge base\nrules; constraints;\nrelationships"),
        (0.38, 0.34, 0.24, 0.32, "Hybrid AI layer\ndata model +\nknowledge score"),
        (0.72, 0.57, 0.25, 0.28, "Explainable output\nopportunity score\nrule trace; factors"),
        (0.72, 0.12, 0.25, 0.34, "Decision use\ntransform; innovate;\nentrepreneurial\naction"),
    ]
    for x, y, w, h, label in boxes:
        face = "#E7F0FA" if "Knowledge" not in label else "#FFF1D6"
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor="#1F3B5B", linewidth=1.5)
        ax.add_patch(rect)
        fontsize = 10.5 if "Hybrid" in label else 9.5
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="#14283D")
    arrows = [
        ((0.26, 0.70), (0.38, 0.53)),
        ((0.26, 0.30), (0.38, 0.47)),
        ((0.62, 0.53), (0.72, 0.70)),
        ((0.845, 0.57), (0.845, 0.46)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#4A6178"})
    ax.text(0.50, 0.90, "Data + domain knowledge", ha="center", fontsize=11, fontweight="bold", color="#1F3B5B")
    ax.text(0.50, 0.25, "Transparent hybrid inference", ha="center", fontsize=10, color="#4A6178")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_results_figure(means: pd.DataFrame, stds: pd.DataFrame, sweep: pd.DataFrame, path: Path) -> None:
    """Write the repeated-seed performance and sensitivity figure."""
    methods = means.index.tolist()
    colors = ["#A7B8C8", "#D9B26F", "#2F6F9F"]
    x = np.arange(len(methods))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.75), dpi=FIGURE_DPI)

    axes[0, 0].bar(x - width / 2, means["ROC-AUC"], width, yerr=stds["ROC-AUC"], capsize=4, color=colors, label="ROC-AUC")
    axes[0, 0].bar(x + width / 2, means["PR-AUC"], width, yerr=stds["PR-AUC"], capsize=4, color=["#D1DCE6", "#E8C98D", "#5B8EB3"], label="PR-AUC")
    axes[0, 0].set_title("Ranking quality")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_xticks(x, methods, rotation=22, ha="right")
    axes[0, 0].set_ylim(0.55, 0.85)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(methods, means["Brier score"], yerr=stds["Brier score"], capsize=4, color=colors)
    axes[0, 1].set_title("Probability calibration")
    axes[0, 1].set_ylabel("Brier score (lower is better)")
    axes[0, 1].tick_params(axis="x", rotation=22)

    axes[1, 0].bar(methods, means["Accuracy"], yerr=stds["Accuracy"], capsize=4, color=colors)
    axes[1, 0].set_title("Classification accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].tick_params(axis="x", rotation=22)
    axes[1, 0].set_ylim(0.55, 0.75)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    line1 = ax.plot(sweep.alpha, sweep["ROC-AUC"], color="#1F5D8C", marker="o", lw=2, label="ROC-AUC")
    line2 = ax2.plot(sweep.alpha, sweep["Brier score"], color="#B5791B", marker="s", lw=2, label="Brier score")
    ax.axvline(0.65, color="#7A7A7A", linestyle="--", lw=1)
    ax.set_title("Blend sensitivity")
    ax.set_xlabel("alpha: data-model weight")
    ax.set_ylabel("ROC-AUC", color="#1F5D8C")
    ax2.set_ylabel("Brier score", color="#B5791B")
    ax.set_ylim(0.70, 0.80)
    ax2.set_ylim(0.19, 0.23)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(line1 + line2, ["ROC-AUC", "Brier score"], frameon=False, fontsize=8, loc="lower left")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Repeated-seed benchmark and hybrid-weight sensitivity", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_correlation_heatmap(data: pd.DataFrame, path: Path) -> None:
    """Write a feature correlation heatmap for the ten indicators and the outcome."""
    columns = FEATURES + ["adopt"]
    corr = data[columns].corr()
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=FIGURE_DPI)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="#14283D")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_feature_importance_figure(runs: list[dict], path: Path) -> None:
    """Write mean +/- sd random-forest feature importance across seeds."""
    importances = np.stack([run["model"].feature_importances_ for run in runs])
    mean_importance = importances.mean(axis=0)
    sd_importance = importances.std(axis=0, ddof=1)
    order = np.argsort(mean_importance)
    features_sorted = [FEATURES[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=FIGURE_DPI)
    ax.barh(
        range(len(features_sorted)),
        mean_importance[order],
        xerr=sd_importance[order],
        color="#2F6F9F",
        capsize=3,
    )
    ax.set_yticks(range(len(features_sorted)), features_sorted, fontsize=9)
    ax.set_xlabel("Mean importance (across seeds)")
    ax.set_title("Random-forest feature importance")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pool(runs: list[dict], key: str) -> np.ndarray:
    """Concatenate a per-run array across all repeated-seed runs."""
    return np.concatenate([run[key] for run in runs])


def make_roc_pr_figure(runs: list[dict], path: Path) -> None:
    """Write pooled ROC and precision-recall curves for all three estimators."""
    y = _pool(runs, "y")
    series = {
        "Data-only AI": _pool(runs, "p_data"),
        "Knowledge rules": _pool(runs, "p_kb"),
        "Hybrid K-AIDE": _pool(runs, "p_hybrid"),
    }
    colors = {"Data-only AI": "#A7B8C8", "Knowledge rules": "#D9B26F", "Hybrid K-AIDE": "#2F6F9F"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), dpi=FIGURE_DPI)
    for name, p in series.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[name], lw=1.8, label=name)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, color=colors[name], lw=1.8, label=name)
    axes[0].plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (pooled)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (pooled)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_confusion_matrix_figure(runs: list[dict], path: Path) -> None:
    """Write pooled confusion matrices at the 0.5 threshold for all three estimators."""
    y = _pool(runs, "y")
    series = {
        "Data-only AI": _pool(runs, "p_data"),
        "Knowledge rules": _pool(runs, "p_kb"),
        "Hybrid K-AIDE": _pool(runs, "p_hybrid"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), dpi=FIGURE_DPI)
    for ax, (name, p) in zip(axes, series.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_yticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Row-normalized rate")
    fig.suptitle("Confusion matrices (pooled, threshold = 0.5)", y=1.04)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_calibration_figure(runs: list[dict], path: Path) -> None:
    """Write a reliability diagram comparing predicted vs. observed frequency."""
    y = _pool(runs, "y")
    series = {
        "Data-only AI": _pool(runs, "p_data"),
        "Knowledge rules": _pool(runs, "p_kb"),
        "Hybrid K-AIDE": _pool(runs, "p_hybrid"),
    }
    colors = {"Data-only AI": "#A7B8C8", "Knowledge rules": "#D9B26F", "Hybrid K-AIDE": "#2F6F9F"}

    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=FIGURE_DPI)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--", label="Perfect calibration")
    for name, p in series.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", color=colors[name], lw=1.8, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3000, help="records per random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--dataset-file", type=Path, default=None, help="combined CSV with seed, record_id, features, and adopt")
    parser.add_argument("--save-dataset", type=Path, default=None, help="write the generated combined benchmark CSV")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="directory for CSV result artifacts")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)

    source_by_seed = None
    if args.dataset_file is not None:
        source_by_seed = load_dataset(args.dataset_file, args.seeds, args.n_samples)
    elif args.save_dataset is not None:
        generated = build_dataset(args.n_samples, args.seeds)
        args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(args.save_dataset, index=False)
        source_by_seed = load_dataset(args.save_dataset, args.seeds, args.n_samples)

    runs = [run_once(args.n_samples, seed, args.alpha, source_by_seed.get(seed) if source_by_seed else None) for seed in args.seeds]
    means, stds = summarize(runs)
    sweep = alpha_sweep(runs, np.linspace(0.0, 1.0, 11))
    print("Repeated-seed mean (standard deviation):")
    for method in means.index:
        values = [f"{means.loc[method, metric]:.4f} ({stds.loc[method, metric]:.4f})" for metric in means.columns]
        print(f"{method}: " + "; ".join(f"{metric}={value}" for metric, value in zip(means.columns, values)))

    p_data = paired_p_value(runs, "Hybrid K-AIDE", "Data-only AI")
    p_rules = paired_p_value(runs, "Hybrid K-AIDE", "Knowledge rules")
    ablation = ablation_summary(runs)
    trace_rate = float(np.mean([run["trace_rate"] for run in runs]))
    write_results(args.results_dir, means, stds, ablation, sweep, runs, args.seeds, p_data, p_rules)
    print("\n2 x 2 component ablation (mean across seeds):")
    print(ablation[["ROC-AUC", "Brier score", "Accuracy"]].round(4).to_string())
    print(f"\nActive knowledge-rule trace rate: {trace_rate:.3f}")
    print(f"Wilcoxon hybrid vs data-only squared-error p-value: {p_data:.4g}")
    print(f"Wilcoxon hybrid vs knowledge-rules squared-error p-value: {p_rules:.4g}")
    print(f"Saved result CSVs to: {args.results_dir}")

    combined = source_by_seed and pd.concat(source_by_seed.values(), ignore_index=True)
    if combined is None or len(combined) == 0:
        combined = build_dataset(args.n_samples, args.seeds)
    # The two manuscript figures are the supplied correlation audit and the
    # pooled ROC/PR comparison. Other diagnostics remain reproducible outputs.
    make_correlation_heatmap(combined, args.output / "figure_1_correlation_heatmap.png")
    make_roc_pr_figure(runs, args.output / "figure_2_roc_pr_curves.png")
    make_feature_importance_figure(runs, args.output / "diagnostic_feature_importance.png")
    make_confusion_matrix_figure(runs, args.output / "diagnostic_confusion_matrices.png")
    make_calibration_figure(runs, args.output / "diagnostic_calibration_curve.png")
    print("Saved figures to:", args.output)


if __name__ == "__main__":
    main()

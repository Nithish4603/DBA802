from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def create_permutation_importance(model, X_test, y_test, out_dir, max_rows=3000, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(X_test) > max_rows:
        sample_idx = X_test.sample(n=max_rows, random_state=seed).index
        X = X_test.loc[sample_idx]
        y = y_test.loc[sample_idx]
    else:
        X, y = X_test, y_test
    result = permutation_importance(model, X, y, scoring="f1", n_repeats=3, random_state=seed, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    top = imp.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (F1 decrease)")
    plt.title("Top Features - Best Model")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=200)
    plt.close()
    return imp

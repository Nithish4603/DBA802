from pathlib import Path
import pandas as pd

TARGET = "label"
DROP_COLS = ["attack_cat"]
CATEGORICAL = ["proto", "service", "state"]


def load_data(data_dir: str | Path, quick: bool = False, seed: int = 42):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train_local.csv")
    test = pd.read_csv(data_dir / "test_local.csv")
    if quick:
        n_train = min(5000, len(train))
        n_test = min(2000, len(test))
        train = train.sample(n=n_train, random_state=seed).reset_index(drop=True)
        test = test.sample(n=n_test, random_state=seed).reset_index(drop=True)
    X_train = train.drop(columns=[TARGET] + DROP_COLS)
    y_train = train[TARGET].astype(int)
    X_test = test.drop(columns=[TARGET] + DROP_COLS)
    y_test = test[TARGET].astype(int)
    meta_test = test[["attack_cat", TARGET]].copy()
    return X_train, y_train, X_test, y_test, meta_test

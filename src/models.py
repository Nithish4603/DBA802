from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def build_preprocessor(X):
    categorical = [c for c in ["proto", "service", "state"] if c in X.columns]
    numerical = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical),
    ])


def build_models(X, seed=42):
    def pipe(model):
        return Pipeline([("preprocessor", build_preprocessor(X)), ("model", model)])
    return {
        "logistic_regression": pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )),
        "xgboost": pipe(XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            n_jobs=-1, random_state=seed
        )),
        "mlp_neural_network": pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
            batch_size=256, learning_rate_init=1e-3, max_iter=60,
            early_stopping=True, validation_fraction=0.15, random_state=seed
        )),
    }

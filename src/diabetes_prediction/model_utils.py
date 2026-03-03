from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing transformer for numeric and categorical columns.

    Args:
        X: Feature dataframe used to infer column types.
    Returns:
        ColumnTransformer with scaling and one-hot encoding.
    """
    categorical_cols = [c for c in ["Gender"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_model_candidates(random_state: int) -> dict[str, object]:
    """Create baseline model candidates with sensible defaults.

    Args:
        random_state: Seed for reproducible estimators.
    Returns:
        Mapping of model names to estimator instances.
    """
    return {
        "LogReg": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def get_tuning_spaces(_: int) -> dict[str, dict[str, object]]:
    """Return hyperparameter search spaces for each model.

    Args:
        _: Unused random state placeholder.
    Returns:
        Mapping of model names to search configurations.
    """
    return {
        "LogReg": {
            "type": "grid",
            "params": {"model__C": [0.1, 1.0, 10.0]},
        },
        "KNN": {
            "type": "grid",
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
            },
        },
        "SVM_RBF": {
            "type": "random",
            "params": {
                "model__C": np.logspace(-2, 2, 8).tolist(),
                "model__gamma": ["scale", "auto"],
            },
        },
        "RandomForest": {
            "type": "random",
            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            },
        },
    }

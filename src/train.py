from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


DEFAULT_DATA = Path("data/processed/diabetes_clean.csv")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "CLASS" not in df.columns:
        raise ValueError("Target column 'CLASS' not found in dataset.")

    df["CLASS"] = df["CLASS"].astype(str).str.strip()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()

    id_cols = [c for c in ["ID", "No_Pation"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    assert_no_missing(df)
    return df


def assert_no_missing(df: pd.DataFrame) -> None:
    na_counts = df.isna().sum()

    empty_counts = pd.Series(0, index=df.columns, dtype="int64")
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        empty_counts.loc[obj_cols] = (
            df[obj_cols].astype(str).apply(lambda s: s.str.strip().eq("")).sum()
        )

    inf_counts = pd.Series(0, index=df.columns, dtype="int64")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        inf_counts.loc[num_cols] = (
            df[num_cols].isin([np.inf, -np.inf]).sum()
        )

    issue_counts = na_counts.add(empty_counts, fill_value=0).add(inf_counts, fill_value=0)
    if issue_counts.sum() > 0:
        bad = issue_counts[issue_counts > 0].sort_values(ascending=False)
        details = ", ".join(f"{col}={int(cnt)}" for col, cnt in bad.items())
        raise ValueError(
            "Missing/invalid values detected. "
            f"Columns with issues: {details}. "
            "Clean the data or re-enable imputing."
        )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in ["Gender"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_model_candidates(random_state: int) -> dict[str, object]:
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


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "accuracy": scores["test_accuracy"].mean(),
        "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
        "f1_macro": scores["test_f1_macro"].mean(),
        "f1_weighted": scores["test_f1_weighted"].mean(),
    }


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, object],
    random_state: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        metrics = evaluate_pipeline_cv(pipeline, X_train, y_train, cv)
        metrics["model"] = name
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        by=["f1_macro", "balanced_accuracy"], ascending=False
    )


def get_tuning_spaces(random_state: int) -> dict[str, dict[str, object]]:
    return {
        "LogReg": {
            "type": "grid",
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            },
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


def tune_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, object],
    random_state: int,
    n_iter: int,
) -> tuple[pd.DataFrame, dict[str, Pipeline], list[dict[str, object]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    spaces = get_tuning_spaces(random_state)
    rows: list[dict[str, object]] = []
    tuned_models: dict[str, Pipeline] = {}
    tuning_details: list[dict[str, object]] = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        space = spaces[name]
        if space["type"] == "grid":
            search = GridSearchCV(
                pipeline,
                param_grid=space["params"],
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=space["params"],
                n_iter=n_iter,
                cv=cv,
                scoring="f1_macro",
                random_state=random_state,
                n_jobs=-1,
            )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        tuned_models[name] = best_pipeline

        metrics = evaluate_pipeline_cv(best_pipeline, X_train, y_train, cv)
        metrics["model"] = name
        rows.append(metrics)

        clean_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        tuning_details.append(
            {
                "model": name,
                "search_type": space["type"],
                "best_score_f1_macro": search.best_score_,
                "best_params": clean_params,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["f1_macro", "balanced_accuracy"], ascending=False
    )
    return results, tuned_models, tuning_details


def plot_confusion_matrix(labels, cm, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diabetes classification models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to cleaned CSV")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Output model directory")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Output reports directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hyper-parameter tuning before final training",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=20,
        help="RandomizedSearchCV iterations (applies to SVM and RandomForest)",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    X = df.drop(columns=["CLASS"])
    y = df["CLASS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    models = make_model_candidates(args.random_state)

    tuning_details = None
    if args.tune:
        cv_results, tuned_models, tuning_details = tune_candidates(
            X_train,
            y_train,
            preprocessor,
            models,
            args.random_state,
            args.tune_n_iter,
        )
        best_name = cv_results.iloc[0]["model"]
        best_pipeline = tuned_models[best_name]
    else:
        cv_results = evaluate_candidates(
            X_train, y_train, preprocessor, models, args.random_state
        )
        best_name = cv_results.iloc[0]["model"]
        best_pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", models[best_name]),
            ]
        )

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "diabetes_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results.to_dict(orient="records"),
        "tuning": tuning_details,
        "test_report": report,
        "labels": labels,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "tuned": args.tune,
        "tune_n_iter": args.tune_n_iter,
    }

    metrics_path = args.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_path = args.reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(labels, cm, cm_path)

    print("Best model:", best_name)
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)
    print("Saved confusion matrix to:", cm_path)


if __name__ == "__main__":
    main()

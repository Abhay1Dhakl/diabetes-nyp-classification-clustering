from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from diabetes_prediction.data_utils import load_data
from diabetes_prediction.model_utils import build_preprocessor, make_model_candidates
from diabetes_prediction.pipeline_utils import (
    build_pipeline,
    evaluate_models,
    make_ros_factory,
    oversample_available,
    tune_models,
)
from diabetes_prediction.plot_utils import plot_confusion_matrix

DEFAULT_DATA = Path("../data/processed/diabetes_clean.csv")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")


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
    tuned_models: dict[str, object] = {}

    if args.tune:
        cv_results, tuned_models, tuning_details = tune_models(
            X_train,
            y_train,
            preprocessor,
            models,
            args.random_state,
            args.tune_n_iter,
        )
    else:
        cv_results = evaluate_models(X_train, y_train, preprocessor, models, args.random_state)

    ros_results_df = pd.DataFrame()
    ros_tuned_results_df = pd.DataFrame()
    ros_tuned_models: dict[str, object] = {}
    ros_tuning_details = None
    ros_factory = None

    if oversample_available():
        ros_factory = make_ros_factory(args.random_state)
        try:
            ros_results_df = evaluate_models(
                X_train,
                y_train,
                preprocessor,
                models,
                args.random_state,
                sampler_factory=ros_factory,
            )
        except Exception:
            ros_results_df = pd.DataFrame()

        if args.tune:
            try:
                ros_tuned_results_df, ros_tuned_models, ros_tuning_details = tune_models(
                    X_train,
                    y_train,
                    preprocessor,
                    models,
                    args.random_state,
                    args.tune_n_iter,
                    sampler_factory=ros_factory,
                )
            except Exception:
                ros_tuned_results_df = pd.DataFrame()

    candidates = []
    if not cv_results.empty:
        df1 = cv_results.copy()
        df1["source"] = "tuned" if args.tune else "class_weight"
        candidates.append(df1)
    if not ros_results_df.empty:
        df2 = ros_results_df.copy()
        df2["source"] = "ros"
        candidates.append(df2)
    if not ros_tuned_results_df.empty:
        df3 = ros_tuned_results_df.copy()
        df3["source"] = "ros_tuned"
        candidates.append(df3)

    if not candidates:
        raise RuntimeError("No model results available to choose best model.")

    combined_candidates = pd.concat(candidates, ignore_index=True)
    combined_candidates = combined_candidates.sort_values("f1_macro", ascending=False)
    best_row = combined_candidates.iloc[0]
    best_name = best_row["model"]
    best_source = best_row["source"]

    if best_source == "tuned":
        best_pipeline = tuned_models[best_name]
    elif best_source == "ros_tuned":
        best_pipeline = ros_tuned_models[best_name]
    elif best_source == "ros":
        if ros_factory is None:
            raise RuntimeError("Oversampling requested but not available")
        best_pipeline = build_pipeline(preprocessor, models[best_name], sampler_factory=ros_factory)
    else:
        best_pipeline = build_pipeline(preprocessor, models[best_name])

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
        "baseline_cv_results": cv_results.to_dict(orient="records") if not cv_results.empty else [],
        "ros_results": ros_results_df.to_dict(orient="records") if not ros_results_df.empty else [],
        "ros_tuned_results": ros_tuned_results_df.to_dict(orient="records") if not ros_tuned_results_df.empty else [],
        "tuning": tuning_details,
        "ros_tuning": ros_tuning_details,
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

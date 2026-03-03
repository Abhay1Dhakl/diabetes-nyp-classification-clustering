from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - optional dependency
    RandomOverSampler = None
    ImbPipeline = None


def oversample_available() -> bool:
    """Check whether oversampling dependencies are available.

    Args:
        None.
    Returns:
        True if imbalanced-learn is installed, otherwise False.
    """
    return RandomOverSampler is not None and ImbPipeline is not None


def make_ros_factory(random_state: int) -> Callable[[], object]:
    """Create a RandomOverSampler factory for pipelines.

    Args:
        random_state: Seed for reproducibility.
    Returns:
        A callable that creates a RandomOverSampler instance.
    """
    if not oversample_available():
        raise RuntimeError("imbalanced-learn is required for oversampling")
    return lambda: RandomOverSampler(random_state=random_state)


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    """Evaluate a pipeline with cross-validation.

    Args:
        pipeline: Model pipeline to evaluate.
        X_train: Training features.
        y_train: Training labels.
        cv: Stratified cross-validation splitter.
    Returns:
        Dictionary of averaged CV metrics.
    """
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


def build_pipeline(
    preprocessor,
    model: object,
    sampler_factory: Callable[[], object] | None = None,
) -> Pipeline:
    """Build a preprocessing + model pipeline, optionally with oversampling.

    Args:
        preprocessor: ColumnTransformer for preprocessing.
        model: Estimator instance.
        sampler_factory: Optional sampler factory for oversampling.
    Returns:
        A scikit-learn or imblearn pipeline instance.
    """
    steps = [("preprocess", preprocessor)]
    if sampler_factory is not None:
        if ImbPipeline is None:
            raise RuntimeError("imbalanced-learn is required for oversampling")
        steps.append(("sampler", sampler_factory()))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)
    steps.append(("model", model))
    return Pipeline(steps=steps)


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    models: dict[str, object],
    random_state: int,
    sampler_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Evaluate a set of candidate models with CV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: ColumnTransformer for preprocessing.
        models: Model name to estimator mapping.
        random_state: Seed for CV shuffling.
        sampler_factory: Optional sampler factory for oversampling.
    Returns:
        DataFrame of averaged CV metrics per model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []

    for name, model in models.items():
        pipeline = build_pipeline(preprocessor, model, sampler_factory)
        metrics = evaluate_pipeline_cv(pipeline, X_train, y_train, cv)
        metrics["model"] = name
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        by=["f1_macro", "balanced_accuracy"], ascending=False
    )

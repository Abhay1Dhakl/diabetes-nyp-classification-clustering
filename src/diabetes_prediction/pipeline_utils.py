from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from diabetes_prediction.model_utils import get_tuning_spaces

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - optional dependency
    RandomOverSampler = None
    ImbPipeline = None


def oversample_available() -> bool:
    return RandomOverSampler is not None and ImbPipeline is not None


def make_ros_factory(random_state: int) -> Callable[[], object]:
    if not oversample_available():
        raise RuntimeError("imbalanced-learn is required for oversampling")
    return lambda: RandomOverSampler(random_state=random_state)


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


def build_pipeline(
    preprocessor,
    model: object,
    sampler_factory: Callable[[], object] | None = None,
) -> Pipeline:
    steps = [("preprocess", preprocessor)]
    if sampler_factory is not None:
        if ImbPipeline is None:
            raise RuntimeError("imbalanced-learn is required for oversampling")
        steps.append(("sampler", sampler_factory()))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)
    steps.append(("model", model))
    return Pipeline(steps=steps)


def build_search(
    pipeline: Pipeline,
    space: dict[str, object],
    cv: StratifiedKFold,
    n_iter: int,
    random_state: int,
) -> GridSearchCV | RandomizedSearchCV:
    if space["type"] == "grid":
        return GridSearchCV(
            pipeline,
            param_grid=space["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
    return RandomizedSearchCV(
        pipeline,
        param_distributions=space["params"],
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
    )


def clean_search_params(params: dict[str, object]) -> dict[str, object]:
    return {
        k.replace("model__", "").replace("sampler__", "sampler_"): v
        for k, v in params.items()
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    models: dict[str, object],
    random_state: int,
    sampler_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
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


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    models: dict[str, object],
    random_state: int,
    n_iter: int,
    sampler_factory: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], list[dict[str, object]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    spaces = get_tuning_spaces(random_state)
    rows: list[dict[str, object]] = []
    tuned_models: dict[str, Pipeline] = {}
    tuning_details: list[dict[str, object]] = []

    for name, model in models.items():
        pipeline = build_pipeline(preprocessor, model, sampler_factory)
        space = spaces[name]
        search = build_search(pipeline, space, cv, n_iter, random_state)
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        tuned_models[name] = best_pipeline

        metrics = evaluate_pipeline_cv(best_pipeline, X_train, y_train, cv)
        metrics["model"] = name
        rows.append(metrics)

        tuning_details.append(
            {
                "model": name,
                "search_type": space["type"],
                "best_score_f1_macro": search.best_score_,
                "best_params": clean_search_params(search.best_params_),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["f1_macro", "balanced_accuracy"], ascending=False
    )
    return results, tuned_models, tuning_details

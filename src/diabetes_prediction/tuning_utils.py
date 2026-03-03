from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from diabetes_prediction.model_utils import get_tuning_spaces
from diabetes_prediction.pipeline_utils import build_pipeline, evaluate_pipeline_cv


def build_search(
    pipeline: Pipeline,
    space: dict[str, object],
    cv: StratifiedKFold,
    n_iter: int,
    random_state: int,
) -> GridSearchCV | RandomizedSearchCV:
    """Create a grid or randomized search object.

    Args:
        pipeline: Pipeline to tune.
        space: Search configuration dict.
        cv: Cross-validation splitter.
        n_iter: Random search iterations.
        random_state: Seed for randomized search.
    Returns:
        A configured GridSearchCV or RandomizedSearchCV instance.
    """
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
    """Normalize tuned parameter names for reporting.

    Args:
        params: Raw parameter dict from a search object.
    Returns:
        Cleaned parameter dict with shorter keys.
    """
    return {
        k.replace("model__", "").replace("sampler__", "sampler_"): v
        for k, v in params.items()
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    models: dict[str, object],
    random_state: int,
    n_iter: int,
    sampler_factory: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], list[dict[str, object]]]:
    """Tune models with CV and return best pipelines.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: ColumnTransformer for preprocessing.
        models: Model name to estimator mapping.
        random_state: Seed for CV and random search.
        n_iter: RandomizedSearchCV iterations.
        sampler_factory: Optional sampler factory for oversampling.
    Returns:
        Tuple of (results DataFrame, tuned models, tuning details).
    """
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

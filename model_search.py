"""Benchmark practical classifiers on the cleaned limitation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from lstm_model import (
    DATA_PATH,
    REPORTS_DIR,
    RANDOM_SEED,
    SPLIT_DATE_COLUMN,
    TARGET_COLUMN,
    build_feature_matrix,
    calculate_metrics,
    chronological_masks,
    get_feature_columns,
    load_data,
    real_row_mask,
    tune_threshold,
)


def optional_boosting_candidates(scale_pos_weight: float) -> dict[str, Any]:
    """Return optional third-party gradient boosting models when installed."""
    candidates: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    except ImportError:
        pass
    return candidates


def model_candidates(scale_pos_weight: float) -> dict[str, Any]:
    """Return practical sklearn classifiers worth testing on this dataset."""
    candidates = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(160, 80),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            random_state=RANDOM_SEED,
        ),
    }
    candidates.update(optional_boosting_candidates(scale_pos_weight))
    return candidates


def positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Return positive-class scores from a fitted sklearn classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_chronological_models(
    data: pd.DataFrame,
    feature_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate candidates with chronological train/validation/test splits."""
    targets = data[TARGET_COLUMN].to_numpy(dtype=int)
    masks = chronological_masks(data[SPLIT_DATE_COLUMN])
    sample_weight = compute_sample_weight("balanced", targets[masks["train"]])
    rows = []

    train_counts = np.bincount(targets[masks["train"]], minlength=2)
    scale_pos_weight = float(train_counts[0] / train_counts[1])

    for model_name, model in model_candidates(scale_pos_weight).items():
        if model_name in {"hist_gradient_boosting", "mlp"}:
            model.fit(
                feature_matrix[masks["train"]],
                targets[masks["train"]],
                sample_weight=sample_weight,
            )
        else:
            model.fit(feature_matrix[masks["train"]], targets[masks["train"]])

        validation_scores = positive_scores(model, feature_matrix[masks["validation"]])
        test_scores = positive_scores(model, feature_matrix[masks["test"]])
        threshold, _ = tune_threshold(targets[masks["validation"]], validation_scores)
        metrics = calculate_metrics(targets[masks["test"]], test_scores, threshold)
        rows.append(
            {
                "evaluation": "chronological",
                "model": model_name,
                "threshold": threshold,
                **metrics,
            }
        )

    return rows


def evaluate_random_split_models(
    data: pd.DataFrame,
    feature_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate candidates with the same broad random split style as old work."""
    targets = data[TARGET_COLUMN].to_numpy(dtype=int)
    real_indices = np.flatnonzero(real_row_mask(data))
    synthetic_indices = np.flatnonzero(~real_row_mask(data))
    train_indices, test_indices = train_test_split(
        real_indices,
        test_size=0.3,
        shuffle=True,
        stratify=targets[real_indices],
        random_state=1,
    )
    train_indices = np.concatenate([train_indices, synthetic_indices])
    rows = []
    train_counts = np.bincount(targets[train_indices], minlength=2)
    scale_pos_weight = float(train_counts[0] / train_counts[1])

    for model_name, model in model_candidates(scale_pos_weight).items():
        if model_name in {"hist_gradient_boosting", "mlp"}:
            sample_weight = compute_sample_weight("balanced", targets[train_indices])
            model.fit(
                feature_matrix[train_indices],
                targets[train_indices],
                sample_weight=sample_weight,
            )
        else:
            model.fit(feature_matrix[train_indices], targets[train_indices])

        test_scores = positive_scores(model, feature_matrix[test_indices])
        fixed_metrics = calculate_metrics(targets[test_indices], test_scores, 0.5)
        tuned_threshold, _ = tune_threshold(targets[test_indices], test_scores)
        tuned_metrics = calculate_metrics(
            targets[test_indices],
            test_scores,
            tuned_threshold,
        )
        rows.append(
            {
                "evaluation": "random_split_fixed_0_5",
                "model": model_name,
                "threshold": 0.5,
                **fixed_metrics,
            }
        )
        rows.append(
            {
                "evaluation": "random_split_best_operating_point",
                "model": model_name,
                "threshold": tuned_threshold,
                **tuned_metrics,
            }
        )

    return rows


def run_search(
    data_path: Path = DATA_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Run candidate benchmark and save the results."""
    reports_dir.mkdir(exist_ok=True)
    data = load_data(data_path)
    source_features = get_feature_columns(data)
    train_mask = chronological_masks(data[SPLIT_DATE_COLUMN])["train"]
    feature_matrix, feature_names = build_feature_matrix(data, source_features, train_mask)

    rows = []
    rows.extend(evaluate_chronological_models(data, feature_matrix))
    rows.extend(evaluate_random_split_models(data, feature_matrix))

    results = pd.DataFrame(rows).sort_values(
        ["evaluation", "f1", "pr_auc"],
        ascending=[True, False, False],
    )
    results.to_csv(reports_dir / "model_search_results.csv", index=False)
    metadata = {
        "dataset": str(data_path),
        "rows": int(len(data)),
        "positives": int(data[TARGET_COLUMN].sum()),
        "source_feature_count": len(source_features),
        "model_feature_count": len(feature_names),
    }
    (reports_dir / "model_search_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    output = run_search()
    print(output.to_string(index=False))

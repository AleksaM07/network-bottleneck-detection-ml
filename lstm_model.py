"""Train and evaluate the improved LSTM limitation model.

This keeps the idea from `legacy_scripts/latest_lstm_anomaly_time_series.ipynb`,
but feeds the model the whole cleaned/anonymized CSV instead of a tiny handpicked
subset. The target is `Limitation`; `observation_date` is used only to create the
chronological split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/data.csv")
REPORTS_DIR = Path("reports")
TARGET_COLUMN = "Limitation"
SPLIT_DATE_COLUMN = "observation_date"
TRAIN_END_DATE = "2024-05-10"
VALIDATION_END_DATE = "2024-05-15"
RANDOM_SEED = 17
SEQUENCE_LENGTH = 5
HASH_BUCKETS = 32
EPOCHS = 30
BATCH_SIZE = 128
LEGACY_COMPARABLE_EPOCHS = 20
LEGACY_COMPARABLE_BATCH_SIZE = 32

SEQUENCE_GROUP_COLUMNS = ["Operator", "System ID", "Server IP Address"]
EXCLUDED_FEATURE_COLUMNS = {TARGET_COLUMN, SPLIT_DATE_COLUMN, "_row_order"}


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    """Set reproducibility controls before TensorFlow is imported."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the final anonymized dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Final dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, SPLIT_DATE_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data.copy()
    data[SPLIT_DATE_COLUMN] = pd.to_datetime(data[SPLIT_DATE_COLUMN], errors="raise")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    if sorted(data[TARGET_COLUMN].dropna().unique().tolist()) != [0, 1]:
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values")

    data["_row_order"] = np.arange(len(data))
    return data.sort_values([SPLIT_DATE_COLUMN, "_row_order"]).reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Use every CSV column except the target and split-only helper fields."""
    return [column for column in data.columns if column not in EXCLUDED_FEATURE_COLUMNS]


def chronological_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    """Create chronological train/validation/test masks."""
    train_end = pd.Timestamp(TRAIN_END_DATE)
    validation_end = pd.Timestamp(VALIDATION_END_DATE)
    return {
        "train": (dates <= train_end).to_numpy(),
        "validation": ((dates > train_end) & (dates <= validation_end)).to_numpy(),
        "test": (dates > validation_end).to_numpy(),
    }


def stable_hash_bucket(value: str, buckets: int = HASH_BUCKETS) -> int:
    """Map a categorical feature value into a deterministic hash bucket."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % buckets


def clean_feature_name(column: str) -> str:
    """Create a compact feature-safe label from a source column name."""
    clean = "".join(character.lower() if character.isalnum() else "_" for character in column)
    return "_".join(part for part in clean.split("_") if part)


def build_feature_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess all CSV features into a compact numeric matrix.

    Numeric columns are median-imputed and standardized using train rows only.
    Categorical columns are compressed into deterministic per-column hash
    buckets, similar in spirit to the old notebook's hashing encoder but with
    less collision between unrelated columns.
    """
    numeric_columns = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    numeric_data = data[numeric_columns].to_numpy(dtype=float)
    train_numeric = numeric_data[train_mask]
    medians = np.nanmedian(train_numeric, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    numeric_data = np.where(np.isnan(numeric_data), medians, numeric_data)

    scaler = StandardScaler()
    scaler.fit(numeric_data[train_mask])
    scaled_numeric = scaler.transform(numeric_data).astype(np.float32)

    hashed_blocks = []
    hashed_feature_names = []
    for column in categorical_columns:
        block = np.zeros((len(data), HASH_BUCKETS), dtype=np.float32)
        values = data[column].fillna("missing").astype(str)
        for row_index, value in enumerate(values):
            bucket = stable_hash_bucket(f"{column}={value}")
            block[row_index, bucket] = 1.0
        hashed_blocks.append(block)
        clean_name = clean_feature_name(column)
        hashed_feature_names.extend(
            f"{clean_name}_hash_{bucket:02d}" for bucket in range(HASH_BUCKETS)
        )

    matrices = [scaled_numeric] if len(numeric_columns) else []
    matrices.extend(hashed_blocks)
    feature_matrix = np.hstack(matrices).astype(np.float32)
    feature_names = numeric_columns + hashed_feature_names
    return feature_matrix, feature_names


def make_sequences(
    data: pd.DataFrame,
    feature_matrix: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """Create past-to-current LSTM sequences within anonymized groups."""
    group_columns = [column for column in SEQUENCE_GROUP_COLUMNS if column in data.columns]
    if not group_columns:
        raise ValueError("Sequence group columns are missing from the dataset")

    ordered = data.copy()
    ordered["_matrix_index"] = np.arange(len(ordered))
    ordered = ordered.sort_values(group_columns + [SPLIT_DATE_COLUMN, "_row_order"])
    ordered = ordered.reset_index(drop=True)

    x_values = feature_matrix[ordered["_matrix_index"].to_numpy()]
    y_values = ordered[TARGET_COLUMN].to_numpy(dtype=int)
    dates = ordered[SPLIT_DATE_COLUMN].reset_index(drop=True)

    sequences = np.zeros(
        (len(ordered), sequence_length, feature_matrix.shape[1]),
        dtype=np.float32,
    )
    targets = np.zeros(len(ordered), dtype=int)

    output_index = 0
    for _, group in ordered.groupby(group_columns, sort=False, dropna=False):
        group_positions = group.index.to_numpy()
        for position_in_group, source_index in enumerate(group_positions):
            start = max(0, position_in_group - sequence_length + 1)
            history_positions = group_positions[start : position_in_group + 1]
            history = x_values[history_positions]
            sequences[output_index, -len(history) :, :] = history
            targets[output_index] = y_values[source_index]
            output_index += 1

    metadata = ordered.drop(columns=["_matrix_index"])
    return sequences, targets, dates, metadata


def class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    """Create balanced class weights without synthetic SMOTE samples."""
    counts = np.bincount(y_train, minlength=2)
    total = counts.sum()
    return {
        class_id: float(total / (2 * count)) if count else 1.0
        for class_id, count in enumerate(counts)
    }


def build_lstm_model(sequence_length: int, n_features: int) -> Any:
    """Build the full-CSV LSTM model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, n_features)),
            layers.Masking(mask_value=0.0),
            layers.LSTM(96, return_sequences=False),
            layers.Dropout(0.25),
            layers.Dense(48, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def build_legacy_comparable_lstm_model(n_features: int) -> Any:
    """Build a one-step LSTM comparable to the historical notebook."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = keras.Sequential(
        [
            layers.Input(shape=(1, n_features)),
            layers.LSTM(100),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def oversample_minority(
    x_values: np.ndarray,
    y_values: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance a binary training set by sampling minority rows with replacement."""
    class_zero = np.flatnonzero(y_values == 0)
    class_one = np.flatnonzero(y_values == 1)
    if len(class_zero) == 0 or len(class_one) == 0:
        return x_values, y_values

    rng = np.random.default_rng(seed)
    if len(class_zero) > len(class_one):
        sampled_minority = rng.choice(class_one, size=len(class_zero), replace=True)
        balanced_indices = np.concatenate([class_zero, sampled_minority])
    else:
        sampled_minority = rng.choice(class_zero, size=len(class_one), replace=True)
        balanced_indices = np.concatenate([sampled_minority, class_one])
    rng.shuffle(balanced_indices)
    return x_values[balanced_indices], y_values[balanced_indices]


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose the threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    rows = []
    for index, threshold in enumerate(thresholds):
        p_value = float(precision[index])
        r_value = float(recall[index])
        f1 = 0.0 if p_value + r_value == 0 else 2 * p_value * r_value / (p_value + r_value)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": p_value,
                "recall": r_value,
                "f1": f1,
            }
        )
    curve = pd.DataFrame(rows)
    if curve.empty:
        return 0.5, curve
    best_row = curve.sort_values("f1", ascending=False).iloc[0]
    return float(best_row["threshold"]), curve


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate positive-class metrics for an imbalanced binary classifier."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def split_summary(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Summarize split sizes and target rates."""
    rows = []
    for split_name, mask in masks.items():
        split_dates = metadata.loc[mask, SPLIT_DATE_COLUMN]
        split_targets = targets[mask]
        rows.append(
            {
                "split": split_name,
                "start_date": str(split_dates.min().date()),
                "end_date": str(split_dates.max().date()),
                "rows": int(mask.sum()),
                "positives": int(split_targets.sum()),
                "positive_rate": float(split_targets.mean()),
            }
        )
    return rows


def save_predictions(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
    scores_by_split: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
) -> None:
    """Save scored rows needed for reporting and error analysis."""
    useful_columns = [
        SPLIT_DATE_COLUMN,
        TARGET_COLUMN,
        "Operator",
        "System ID",
        "Server IP Address",
        "Transfer Status",
        "Connection Initiation Status",
        "Transfer Throughput [kbit/s]",
    ]
    frames = []
    for split_name, mask in masks.items():
        selected = [column for column in useful_columns if column in metadata.columns]
        scored = metadata.loc[mask, selected].copy()
        scored[TARGET_COLUMN] = targets[mask]
        scored["split"] = split_name
        scored["score"] = scores_by_split[split_name]
        scored["prediction"] = (scored["score"] >= threshold).astype(int)
        frames.append(scored)
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)


def save_feature_signal(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    feature_names: list[str],
    reports_dir: Path,
) -> None:
    """Save a simple feature signal ranking for the report."""
    current_step = sequences[train_mask, -1, :]
    train_targets = targets[train_mask]

    positive_mean = current_step[train_targets == 1].mean(axis=0)
    negative_mean = current_step[train_targets == 0].mean(axis=0)
    signal = np.abs(positive_mean - negative_mean)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": signal,
            "importance_std": 0.0,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)


def run_legacy_comparable_lstm(
    data: pd.DataFrame,
    source_feature_columns: list[str],
    reports_dir: Path,
) -> dict[str, Any]:
    """Run a random-split, one-step LSTM comparable to the old notebook.

    This is intentionally not the scientific deployment estimate. It exists so
    the current cleaned dataset can be compared against the historical notebook
    setup instead of mixing that result with chronological validation.
    """
    import tensorflow as tf
    from tensorflow import keras

    targets = data[TARGET_COLUMN].to_numpy(dtype=int)
    row_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=0.3,
        shuffle=True,
        random_state=1,
        stratify=targets,
    )

    train_mask = np.zeros(len(data), dtype=bool)
    test_mask = np.zeros(len(data), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    feature_matrix, feature_names = build_feature_matrix(
        data,
        source_feature_columns,
        train_mask,
    )
    x_train = feature_matrix[train_indices].reshape(len(train_indices), 1, -1)
    y_train = targets[train_indices]
    x_test = feature_matrix[test_indices].reshape(len(test_indices), 1, -1)
    y_test = targets[test_indices]
    x_train_balanced, y_train_balanced = oversample_minority(x_train, y_train)

    model = build_legacy_comparable_lstm_model(len(feature_names))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train_balanced,
        y_train_balanced,
        validation_data=(x_test, y_test),
        epochs=LEGACY_COMPARABLE_EPOCHS,
        batch_size=LEGACY_COMPARABLE_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    test_scores = model.predict(
        x_test, batch_size=LEGACY_COMPARABLE_BATCH_SIZE, verbose=0
    ).reshape(-1)
    fixed_threshold = 0.5
    tuned_threshold, threshold_curve = tune_threshold(y_test, test_scores)
    fixed_metrics = calculate_metrics(y_test, test_scores, fixed_threshold)
    tuned_metrics = calculate_metrics(y_test, test_scores, tuned_threshold)

    predictions = data.iloc[test_indices][
        [
            column
            for column in [
                SPLIT_DATE_COLUMN,
                TARGET_COLUMN,
                "Operator",
                "System ID",
                "Server IP Address",
                "Transfer Status",
                "Connection Initiation Status",
                "Transfer Throughput [kbit/s]",
            ]
            if column in data.columns
        ]
    ].copy()
    predictions["split"] = "random_test"
    predictions["score"] = test_scores
    predictions["prediction"] = (predictions["score"] >= fixed_threshold).astype(int)
    predictions.to_csv(reports_dir / "legacy_comparable_predictions.csv", index=False)
    threshold_curve.to_csv(
        reports_dir / "legacy_comparable_threshold_curve.csv", index=False
    )
    pd.DataFrame(history.history).to_csv(
        reports_dir / "legacy_comparable_training_history.csv", index=False
    )
    model.save(reports_dir / "legacy_comparable_lstm_model.keras")

    result = {
        "dataset": str(DATA_PATH),
        "model": "Notebook-comparable full CSV LSTM",
        "feature_set": "all anonymized CSV columns except target/date bookkeeping",
        "split": "stratified random 70/30, notebook-style comparison",
        "sequence_length": 1,
        "hash_buckets_per_categorical_column": HASH_BUCKETS,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "fixed_threshold": fixed_threshold,
        "tuned_threshold_on_random_test": tuned_threshold,
        "metrics": {
            "test_fixed_threshold_0_5": fixed_metrics,
            "test": tuned_metrics,
        },
        "source_feature_count": len(source_feature_columns),
        "model_feature_count": len(feature_names),
        "tensorflow_version": tf.__version__,
    }
    (reports_dir / "legacy_comparable_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def run_model(
    data_path: Path = DATA_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Run the complete full-CSV LSTM flow and write reusable outputs."""
    set_reproducibility()

    import tensorflow as tf
    from tensorflow import keras

    reports_dir.mkdir(exist_ok=True)

    data = load_data(data_path)
    source_feature_columns = get_feature_columns(data)
    row_masks = chronological_masks(data[SPLIT_DATE_COLUMN])
    feature_matrix, feature_names = build_feature_matrix(
        data,
        source_feature_columns,
        row_masks["train"],
    )
    sequences, targets, dates, metadata = make_sequences(data, feature_matrix)
    masks = chronological_masks(dates)

    model = build_lstm_model(SEQUENCE_LENGTH, len(feature_names))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        sequences[masks["train"]],
        targets[masks["train"]],
        validation_data=(sequences[masks["validation"]], targets[masks["validation"]]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict(targets[masks["train"]]),
        callbacks=callbacks,
        verbose=2,
    )

    validation_scores = model.predict(
        sequences[masks["validation"]], batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    threshold, threshold_curve = tune_threshold(targets[masks["validation"]], validation_scores)
    scores_by_split = {
        split_name: model.predict(sequences[mask], batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        for split_name, mask in masks.items()
    }
    metrics = {
        split_name: calculate_metrics(targets[mask], scores_by_split[split_name], threshold)
        for split_name, mask in masks.items()
    }

    threshold_curve.to_csv(reports_dir / "threshold_curve.csv", index=False)
    pd.DataFrame(history.history).to_csv(reports_dir / "training_history.csv", index=False)
    save_predictions(
        metadata,
        targets,
        masks,
        scores_by_split,
        threshold,
        reports_dir / "predictions.csv",
    )
    save_feature_signal(sequences, targets, masks["train"], feature_names, reports_dir)
    model.save(reports_dir / "lstm_model.keras")
    legacy_comparable = run_legacy_comparable_lstm(
        data,
        source_feature_columns,
        reports_dir,
    )

    result = {
        "dataset": str(data_path),
        "rows": int(len(data)),
        "positives": int(data[TARGET_COLUMN].sum()),
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "model": "Full CSV LSTM",
        "feature_set": "all anonymized CSV columns except target/date bookkeeping",
        "sequence_length": SEQUENCE_LENGTH,
        "hash_buckets_per_categorical_column": HASH_BUCKETS,
        "threshold": threshold,
        "random_seed": RANDOM_SEED,
        "split_summary": split_summary(metadata, targets, masks),
        "metrics": metrics,
        "legacy_comparable": legacy_comparable,
        "source_feature_count": len(source_feature_columns),
        "model_feature_count": len(feature_names),
        "source_features": source_feature_columns,
        "features": feature_names,
        "tensorflow_version": tf.__version__,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    summary = run_model()
    test_metrics = summary["metrics"]["test"]
    print(
        "Test metrics: "
        f"precision={test_metrics['precision']:.3f}, "
        f"recall={test_metrics['recall']:.3f}, "
        f"f1={test_metrics['f1']:.3f}, "
        f"pr_auc={test_metrics['pr_auc']:.3f}, "
        f"roc_auc={test_metrics['roc_auc']:.3f}"
    )

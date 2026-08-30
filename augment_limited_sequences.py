"""Create conservative train-only augmentation from real limited sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/data_with_sequence.csv")
OUTPUT_PATH = Path("data/data_augmented.csv")
REPORT_PATH = Path("reports/augmentation_summary.csv")
TARGET_COLUMN = "Limitation"
DATE_COLUMN = "observation_date"
TRAIN_END_DATE = "2024-05-10"
RANDOM_SEED = 17
SYNTHETIC_LIMITED_ROWS = 10_000
MIN_CONSECUTIVE_LIMITED_ROWS = 3

EPISODE_GROUP_COLUMNS = [
    DATE_COLUMN,
    "Operator",
    "System ID",
    "Route Name",
    "Server IP Address",
]

JITTER_RULES = {
    "throughput": {
        "contains": ["Throughput", "throughput"],
        "low": 0.92,
        "high": 1.08,
        "row_noise": 0.015,
    },
    "rtt": {
        "contains": ["RTT"],
        "low": 0.95,
        "high": 1.10,
        "row_noise": 0.02,
    },
    "duration": {
        "contains": ["Duration"],
        "low": 0.94,
        "high": 1.08,
        "row_noise": 0.02,
    },
    "radio": {
        "contains": ["RSRP", "RSRQ", "SINR", "BLER"],
        "low": 0.97,
        "high": 1.03,
        "row_noise": 0.01,
    },
    "bandwidth_usage": {
        "contains": ["Bandwidth", "Usage", "spectral efficiency"],
        "low": 0.94,
        "high": 1.06,
        "row_noise": 0.015,
    },
}

PROTECTED_COLUMNS = {
    TARGET_COLUMN,
    DATE_COLUMN,
    "Date",
    "Sequence",
    "sequence_in_day_rank",
    "sequence_in_day_pct",
    "sequence_delta_previous",
    "time_delta_previous_seconds",
}


def numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric view of a column."""
    return pd.to_numeric(data[column], errors="coerce")


def q(series: pd.Series, quantile: float) -> float:
    """Return a finite quantile or NaN when unavailable."""
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan
    return float(finite.quantile(quantile))


def sort_columns(data: pd.DataFrame) -> list[str]:
    """Return stable sequence order columns."""
    columns = [column for column in EPISODE_GROUP_COLUMNS if column in data.columns]
    for column in ["Sequence", "network_access_start_seconds"]:
        if column in data.columns:
            columns.append(column)
    return columns


def good_condition_mask(data: pd.DataFrame) -> pd.Series:
    """Mark rows where non-Akamai network conditions look usable enough."""
    baseline = data[data[TARGET_COLUMN].eq(0)]
    checks: list[pd.Series] = []

    if "Average Used Bandwidth DL" in data.columns:
        threshold = q(baseline["Average Used Bandwidth DL"], 0.15)
        if np.isfinite(threshold):
            checks.append(numeric(data, "Average Used Bandwidth DL").ge(threshold))
    if "Maximum Available Bandwidth DL" in data.columns:
        threshold = q(baseline["Maximum Available Bandwidth DL"], 0.15)
        if np.isfinite(threshold):
            checks.append(numeric(data, "Maximum Available Bandwidth DL").ge(threshold))
    if "E2E spectral efficiency DL" in data.columns:
        threshold = q(baseline["E2E spectral efficiency DL"], 0.10)
        if np.isfinite(threshold):
            checks.append(numeric(data, "E2E spectral efficiency DL").ge(threshold))
    if "LTE PCC SINR Avg" in data.columns:
        checks.append(numeric(data, "LTE PCC SINR Avg").ge(3.0))
    if "NR PCell SSB Serving Beam SINR Avg" in data.columns:
        checks.append(numeric(data, "NR PCell SSB Serving Beam SINR Avg").ge(3.0))
    if "LTE PCC RSRP Avg" in data.columns:
        checks.append(numeric(data, "LTE PCC RSRP Avg").ge(-105.0))
    if "NR PCell SSB Serving Beam RSRP Avg" in data.columns:
        checks.append(numeric(data, "NR PCell SSB Serving Beam RSRP Avg").ge(-105.0))

    if not checks:
        return pd.Series(True, index=data.index)

    score = pd.concat(checks, axis=1).fillna(False).mean(axis=1)
    return score.ge(0.5)


def find_limited_episodes(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Find train-only positive episodes with at least three adjacent rows."""
    train_end = pd.Timestamp(TRAIN_END_DATE)
    working = data.copy()
    working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN], errors="raise")
    working["_good_conditions"] = good_condition_mask(working)

    train = working[
        working[DATE_COLUMN].le(train_end)
        & working[TARGET_COLUMN].eq(1)
        & working["_good_conditions"]
    ].copy()
    if train.empty:
        return []

    groups = [column for column in EPISODE_GROUP_COLUMNS if column in train.columns]
    episodes = []
    ordered = train.sort_values(sort_columns(train))
    for _, group in ordered.groupby(groups, dropna=False, sort=False):
        if len(group) < MIN_CONSECUTIVE_LIMITED_ROWS:
            continue
        sequence_delta = numeric(group, "Sequence").diff().fillna(0)
        new_run = sequence_delta.gt(sequence_delta[sequence_delta.gt(0)].median() * 3)
        run_id = new_run.cumsum()
        for _, episode in group.groupby(run_id, sort=False):
            if len(episode) >= MIN_CONSECUTIVE_LIMITED_ROWS:
                if "Server IP Address" in episode.columns:
                    server_count = episode["Server IP Address"].nunique(dropna=False)
                    if server_count != 1:
                        continue
                episodes.append(episode.drop(columns=["_good_conditions"]))
    return episodes


def jitter_multiplier(rng: np.random.Generator, rule: dict[str, Any], rows: int) -> np.ndarray:
    """Create a shared episode multiplier plus small row noise."""
    base = rng.uniform(float(rule["low"]), float(rule["high"]))
    row_noise = rng.normal(1.0, float(rule["row_noise"]), size=rows)
    return base * row_noise


def jitter_numeric_columns(
    synthetic: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Slightly vary KPI values while preserving the limitation-wall shape."""
    output = synthetic.copy()
    numeric_columns = output.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column in PROTECTED_COLUMNS or column.startswith("_"):
            continue

        matched_rule = None
        for rule in JITTER_RULES.values():
            if any(token in column for token in rule["contains"]):
                matched_rule = rule
                break
        if matched_rule is None:
            continue

        values = pd.to_numeric(output[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        multiplier = jitter_multiplier(rng, matched_rule, len(output))
        output[column] = values * multiplier

    if "network_access_start_seconds" in output.columns:
        seconds = numeric(output, "network_access_start_seconds")
        output["network_access_start_seconds"] = seconds + rng.normal(0, 2.0, len(output))
    if "Sequence" in output.columns:
        sequence = numeric(output, "Sequence")
        output["Sequence"] = sequence + rng.normal(0, 0.05, len(output))

    output[TARGET_COLUMN] = 1
    output["_is_synthetic_limited"] = 1
    return output


def create_synthetic_rows(
    episodes: list[pd.DataFrame],
    target_rows: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample real positive episodes and create conservative synthetic copies."""
    if not episodes:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    generated = []
    generated_rows = 0
    while generated_rows < target_rows:
        episode = episodes[int(rng.integers(0, len(episodes)))]
        max_window = min(len(episode), 7)
        window_size = int(rng.integers(MIN_CONSECUTIVE_LIMITED_ROWS, max_window + 1))
        start = int(rng.integers(0, len(episode) - window_size + 1))
        window = episode.iloc[start : start + window_size].copy()
        synthetic = jitter_numeric_columns(window, rng)
        synthetic["_synthetic_episode_id"] = len(generated)
        generated.append(synthetic)
        generated_rows += len(synthetic)

    return pd.concat(generated, ignore_index=True).head(target_rows)


def augment_limited_sequences(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
) -> pd.DataFrame:
    """Write an augmented dataset with train-only synthetic positive sequences."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run add_sequence_features.py once first."
        )

    data = pd.read_csv(input_path)
    data["_is_synthetic_limited"] = 0
    data["_synthetic_episode_id"] = pd.NA

    episodes = find_limited_episodes(data)
    synthetic = create_synthetic_rows(episodes, SYNTHETIC_LIMITED_ROWS)
    augmented = pd.concat([data, synthetic], ignore_index=True)
    augmented[DATE_COLUMN] = pd.to_datetime(augmented[DATE_COLUMN]).dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(exist_ok=True)
    report_path.parent.mkdir(exist_ok=True)
    augmented.to_csv(output_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "input_rows": len(data),
                "input_positive_rows": int(data[TARGET_COLUMN].sum()),
                "eligible_positive_episodes": len(episodes),
                "single_ip_positive_episodes": len(episodes),
                "min_consecutive_limited_rows": MIN_CONSECUTIVE_LIMITED_ROWS,
                "synthetic_positive_rows": len(synthetic),
                "output_rows": len(augmented),
                "output_positive_rows": int(augmented[TARGET_COLUMN].sum()),
                "train_end_date": TRAIN_END_DATE,
            }
        ]
    )
    summary.to_csv(report_path, index=False)
    return summary


if __name__ == "__main__":
    result = augment_limited_sequences()
    print(result.to_string(index=False))

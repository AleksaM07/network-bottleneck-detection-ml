"""Create one versioned dashboard image from model outputs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


REPORTS_DIR = Path("reports")
REPORT_NAME_PATTERN = re.compile(r"report-v(\d+)\.png$")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _next_report_path(reports_dir: Path) -> Path:
    versions = []
    for path in reports_dir.glob("report-v*.png"):
        match = REPORT_NAME_PATTERN.fullmatch(path.name)
        if match:
            versions.append(int(match.group(1)))
    next_version = max(versions, default=0) + 1
    return reports_dir / f"report-v{next_version}.png"


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", pad=10)
    axis.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)


def _metric_table_text(
    metrics: dict[str, Any],
    model_search: pd.DataFrame,
    sequence_summary: pd.DataFrame,
) -> str:
    test = metrics["metrics"]["test"]
    comparable = metrics.get("legacy_comparable", {})
    comparable_fixed = comparable.get("metrics", {}).get("test_fixed_threshold_0_5", {})
    lines = [
        f"Dataset: {metrics['dataset']}",
        f"Rows: {metrics['rows']:,}",
        f"Positive rows: {metrics['positives']:,} ({metrics['positive_rate']:.2%})",
        f"LSTM threshold: {metrics['threshold']:.3f}",
        "",
        "Chronological LSTM",
        f"Precision {test['precision']:.3f} | Recall {test['recall']:.3f}",
        f"F1 {test['f1']:.3f} | ROC-AUC {test['roc_auc']:.3f}",
    ]

    if not sequence_summary.empty:
        sequence = sequence_summary.iloc[0]
        lines.extend(
            [
                "",
                "Recovered sequence",
                f"Rows {int(sequence['matched_rows']):,} / "
                f"{int(sequence['active_rows']):,} "
                f"({float(sequence['matched_rate']):.1%})",
                f"Positive rows {int(sequence['matched_positive_rows']):,} / "
                f"{int(sequence['positive_rows']):,} "
                f"({float(sequence['matched_positive_rate']):.1%})",
            ]
        )

    if comparable_fixed:
        lines.extend(
            [
                "",
                "Notebook-style LSTM",
                f"Precision {comparable_fixed['precision']:.3f} | "
                f"Recall {comparable_fixed['recall']:.3f}",
                f"F1 {comparable_fixed['f1']:.3f} | "
                f"ROC-AUC {comparable_fixed['roc_auc']:.3f}",
            ]
        )

    if not model_search.empty:
        for evaluation, label in [
            ("chronological", "Best chronological ML"),
            ("random_split_fixed_0_5", "Best notebook-style ML"),
        ]:
            subset = model_search[model_search["evaluation"] == evaluation]
            if subset.empty:
                continue
            best = subset.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
            lines.extend(
                [
                    "",
                    label,
                    f"{best['model']}: F1 {best['f1']:.3f}, "
                    f"P {best['precision']:.3f}, R {best['recall']:.3f}",
                ]
            )
    return "\n".join(lines)


def _plot_summary_card(
    axis: plt.Axes,
    metrics: dict[str, Any],
    model_search: pd.DataFrame,
    sequence_summary: pd.DataFrame,
) -> None:
    axis.axis("off")
    axis.text(
        0.02,
        0.98,
        _metric_table_text(metrics, model_search, sequence_summary),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    axis.set_title("Run Summary", fontsize=12, fontweight="bold", pad=10)


def _plot_metric_bars(axis: plt.Axes, metrics: dict[str, Any]) -> None:
    rows = []
    for split in ["train", "validation", "test"]:
        split_metrics = metrics["metrics"][split]
        for metric in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            rows.append(
                {
                    "split": split,
                    "metric": metric.upper().replace("_", "-"),
                    "value": split_metrics[metric],
                }
            )
    data = pd.DataFrame(rows)
    pivot = data.pivot(index="metric", columns="split", values="value")
    pivot.loc[["PRECISION", "RECALL", "F1", "PR-AUC", "ROC-AUC"]].plot(
        kind="bar",
        ax=axis,
        width=0.78,
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis, "Precision / Recall / F1 / AUC")


def _plot_confusion(axis: plt.Axes, metrics: dict[str, Any]) -> None:
    test = metrics["metrics"]["test"]
    matrix = np.array(
        [
            [test["true_negatives"], test["false_positives"]],
            [test["false_negatives"], test["true_positives"]],
        ]
    )
    image = axis.imshow(matrix, cmap="Blues")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    axis.set_title("Confusion Matrix: Test", fontsize=12, fontweight="bold", pad=10)


def _plot_class_outcome(
    axis: plt.Axes,
    correct: int,
    wrong: int,
    title: str,
) -> None:
    """Plot one class separately so minority-class errors stay readable."""
    total = correct + wrong
    values = [correct, wrong]
    labels = ["Correct", "Wrong"]
    bars = axis.bar(labels, values, color=["#54A24B", "#E45756"], width=0.55)
    axis.set_ylabel("Rows")
    y_max = max(values) * 1.18 if max(values) else 1
    axis.set_ylim(0, y_max)

    for bar, value in zip(bars, values):
        share = 0.0 if total == 0 else value / total
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.025,
            f"{value:,}\n{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    _style_axis(axis, title)


def _plot_pr_curve(axis: plt.Axes, predictions: pd.DataFrame) -> None:
    precision, recall, _ = precision_recall_curve(
        predictions["Limitation"],
        predictions["score"],
    )
    pr_auc = auc(recall, precision)
    axis.plot(recall, precision, color="#4C78A8", linewidth=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    _style_axis(axis, f"Precision-Recall Curve (AUC {pr_auc:.3f})")


def _plot_roc_curve(axis: plt.Axes, predictions: pd.DataFrame) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        predictions["Limitation"],
        predictions["score"],
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)
    axis.plot(false_positive_rate, true_positive_rate, color="#F58518", linewidth=2)
    axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    _style_axis(axis, f"ROC Curve (AUC {roc_auc:.3f})")


def _plot_threshold_curve(axis: plt.Axes, threshold_curve: pd.DataFrame) -> None:
    if threshold_curve.empty:
        axis.axis("off")
        axis.set_title("Threshold Curve Missing", fontsize=12, fontweight="bold")
        return
    step = max(len(threshold_curve) // 1200, 1)
    sampled = threshold_curve.iloc[::step]
    axis.plot(sampled["threshold"], sampled["precision"], label="Precision", color="#4C78A8")
    axis.plot(sampled["threshold"], sampled["recall"], label="Recall", color="#F58518")
    axis.plot(sampled["threshold"], sampled["f1"], label="F1", color="#54A24B")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis, "Threshold Trade-off")


def _plot_model_search(axis: plt.Axes, model_search: pd.DataFrame) -> None:
    if model_search.empty:
        axis.axis("off")
        axis.set_title("Model Search Missing", fontsize=12, fontweight="bold")
        return
    selected = model_search[
        model_search["evaluation"].isin(["chronological", "random_split_fixed_0_5"])
    ].copy()
    selected["label"] = selected["evaluation"].map(
        {
            "chronological": "chrono",
            "random_split_fixed_0_5": "random",
        }
    ) + " / " + selected["model"]
    selected = selected.sort_values("f1").tail(12)
    axis.barh(selected["label"], selected["f1"], color="#72B7B2")
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("F1")
    _style_axis(axis, "Model Search F1")


def _plot_feature_signal(axis: plt.Axes, importance: pd.DataFrame) -> None:
    if importance.empty:
        axis.axis("off")
        axis.set_title("Feature Signal Missing", fontsize=12, fontweight="bold")
        return
    top = importance.head(12).iloc[::-1]
    axis.barh(top["feature"], top["importance_mean"], color="#B279A2")
    axis.set_xlabel("Class mean gap")
    _style_axis(axis, "Top Feature Signals")


def _plot_errors_by_date(axis: plt.Axes, predictions: pd.DataFrame) -> None:
    data = predictions.copy()
    data["error_type"] = "TN"
    data.loc[(data["Limitation"] == 1) & (data["prediction"] == 1), "error_type"] = "TP"
    data.loc[(data["Limitation"] == 0) & (data["prediction"] == 1), "error_type"] = "FP"
    data.loc[(data["Limitation"] == 1) & (data["prediction"] == 0), "error_type"] = "FN"
    grouped = (
        data.groupby(["observation_date", "error_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for column in ["TP", "FP", "FN"]:
        if column not in grouped:
            grouped[column] = 0
    grouped[["TP", "FP", "FN"]].plot(
        ax=axis,
        marker="o",
        color=["#54A24B", "#E45756", "#F58518"],
    )
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylabel("Rows")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis, "Test Errors by Date")


def _cleanup_intermediate_files(reports_dir: Path, keep_path: Path) -> None:
    """Keep only versioned dashboard images in reports/."""
    for path in reports_dir.iterdir():
        if path.resolve() == keep_path.resolve():
            continue
        if REPORT_NAME_PATTERN.fullmatch(path.name):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def make_report(
    reports_dir: Path = REPORTS_DIR,
    cleanup_intermediate_files: bool = True,
) -> Path:
    """Generate a single versioned dashboard image."""
    reports_dir.mkdir(exist_ok=True)
    metrics = _load_json(reports_dir / "metrics.json")
    predictions = _load_csv(reports_dir / "predictions.csv")
    if predictions.empty:
        raise FileNotFoundError(f"Model predictions not found: {reports_dir / 'predictions.csv'}")
    test_predictions = predictions[predictions["split"] == "test"].copy()
    model_search = _load_csv(reports_dir / "model_search_results.csv")
    threshold_curve = _load_csv(reports_dir / "threshold_curve.csv")
    importance = _load_csv(reports_dir / "feature_importance.csv")
    sequence_summary = _load_csv(reports_dir / "sequence_feature_merge_summary.csv")
    augmentation_summary = _load_csv(reports_dir / "augmentation_summary.csv")

    figure, axes = plt.subplots(5, 2, figsize=(22, 30))
    axes = axes.ravel()
    figure.suptitle(
        "Network Bottleneck Limitation Detection - Report",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )
    if not augmentation_summary.empty:
        augmentation = augmentation_summary.iloc[0]
        figure.text(
            0.5,
            0.981,
            "Train augmentation: "
            f"+{int(augmentation['synthetic_positive_rows']):,} synthetic limited rows "
            f"from {int(augmentation['eligible_positive_episodes'])} real episodes "
            f"(min {int(augmentation['min_consecutive_limited_rows'])} consecutive rows)",
            ha="center",
            va="top",
            fontsize=11,
        )

    test = metrics["metrics"]["test"]
    _plot_metric_bars(axes[0], metrics)
    _plot_confusion(axes[1], metrics)
    _plot_class_outcome(
        axes[2],
        correct=test["true_negatives"],
        wrong=test["false_positives"],
        title="Actual Class 0: Correct vs Wrong",
    )
    _plot_class_outcome(
        axes[3],
        correct=test["true_positives"],
        wrong=test["false_negatives"],
        title="Actual Class 1: Correct vs Wrong",
    )
    _plot_pr_curve(axes[4], test_predictions)
    _plot_roc_curve(axes[5], test_predictions)
    _plot_threshold_curve(axes[6], threshold_curve)
    _plot_model_search(axes[7], model_search)
    _plot_feature_signal(axes[8], importance)
    _plot_errors_by_date(axes[9], test_predictions)

    report_path = _next_report_path(reports_dir)
    figure.tight_layout(rect=[0, 0, 1, 0.972])
    figure.savefig(report_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if cleanup_intermediate_files:
        _cleanup_intermediate_files(reports_dir, report_path)
    return report_path


if __name__ == "__main__":
    output_path = make_report()
    print(f"Report image written to {output_path}")

"""Create compact evaluation reports from model outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


REPORTS_DIR = Path("reports")


def _load_metrics(reports_dir: Path) -> dict[str, Any]:
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Model metrics not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _plot_curves(test_predictions: pd.DataFrame, reports_dir: Path) -> None:
    y_true = test_predictions["Limitation"]
    scores = test_predictions["score"]

    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.title("Precision-Recall Curve: Test Split")
    plt.tight_layout()
    plt.savefig(reports_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, scores)
    plt.title("ROC Curve: Test Split")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()


def _plot_confusion_matrix(test_predictions: pd.DataFrame, reports_dir: Path) -> None:
    y_true = test_predictions["Limitation"]
    predictions = test_predictions["prediction"]

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No limitation", "Limitation"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix: Test Split")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    pd.crosstab(
        y_true,
        predictions,
        rownames=["actual"],
        colnames=["predicted"],
    ).to_csv(reports_dir / "confusion_matrix.csv")


def _write_error_slices(predictions: pd.DataFrame, reports_dir: Path) -> None:
    scored = predictions.copy()
    scored["error_type"] = "true_negative"
    scored.loc[
        (scored["Limitation"] == 1) & (scored["prediction"] == 1), "error_type"
    ] = "true_positive"
    scored.loc[
        (scored["Limitation"] == 0) & (scored["prediction"] == 1), "error_type"
    ] = "false_positive"
    scored.loc[
        (scored["Limitation"] == 1) & (scored["prediction"] == 0), "error_type"
    ] = "false_negative"

    for column in ["observation_date", "Operator", "Transfer Status", "Connection Initiation Status"]:
        if column not in scored.columns:
            continue
        grouped = (
            scored[scored["split"] == "test"]
            .groupby(column, dropna=False)
            .agg(
                rows=("Limitation", "size"),
                positives=("Limitation", "sum"),
                predicted_positives=("prediction", "sum"),
                false_positives=("error_type", lambda values: int((values == "false_positive").sum())),
                false_negatives=("error_type", lambda values: int((values == "false_negative").sum())),
                mean_score=("score", "mean"),
            )
            .reset_index()
            .sort_values(["false_negatives", "false_positives", "rows"], ascending=False)
        )
        clean_name = column.lower().replace(" ", "_").replace("/", "_")
        grouped.to_csv(reports_dir / f"errors_by_{clean_name}.csv", index=False)


def _plot_feature_importance(reports_dir: Path) -> None:
    importance_path = reports_dir / "feature_importance.csv"
    if not importance_path.exists():
        return

    importance = pd.read_csv(importance_path).head(15)
    if importance.empty:
        return

    plt.figure(figsize=(9, 6))
    plt.barh(importance["feature"][::-1], importance["importance_mean"][::-1])
    plt.xlabel("Feature signal, absolute class mean gap")
    plt.title("Top Feature Signals")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png", dpi=150)
    plt.close()


def _write_summary(metrics: dict[str, Any], reports_dir: Path) -> None:
    test = metrics["metrics"]["test"]
    comparable = metrics.get("legacy_comparable", {})
    comparable_fixed = comparable.get("metrics", {}).get("test_fixed_threshold_0_5", {})
    comparable_best = comparable.get("metrics", {}).get("test", {})
    lines = [
        "# Model Evaluation Summary",
        "",
        f"Dataset: `{metrics['dataset']}`",
        f"Rows: {metrics['rows']:,}",
        f"Positive rows: {metrics['positives']:,}",
        f"Model: {metrics['model']}",
        f"Feature set: {metrics['feature_set']}",
        f"Validation-selected threshold: {metrics['threshold']:.6f}",
        "",
        "## Test Metrics",
        "",
        f"- Precision: {test['precision']:.3f}",
        f"- Recall: {test['recall']:.3f}",
        f"- F1: {test['f1']:.3f}",
        f"- PR-AUC: {test['pr_auc']:.3f}",
        f"- ROC-AUC: {test['roc_auc']:.3f}",
        f"- Confusion matrix: TN={test['true_negatives']}, FP={test['false_positives']}, "
        f"FN={test['false_negatives']}, TP={test['true_positives']}",
    ]

    importance_path = reports_dir / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(10)
        if not importance.empty:
            lines.extend(["", "## Top Feature Signals", ""])
            for _, row in importance.iterrows():
                lines.append(
                    f"- {row['feature']}: {row['importance_mean']:.3f}"
                )

    if comparable_fixed:
        lines.extend(
            [
                "",
                "## Notebook-Comparable Random Split",
                "",
                "This is not the final scientific estimate. It is included to compare",
                "against the historical notebook's random-split LSTM result.",
                "",
                "At fixed threshold 0.5, matching the old notebook style:",
                "",
                f"- Precision: {comparable_fixed['precision']:.3f}",
                f"- Recall: {comparable_fixed['recall']:.3f}",
                f"- F1: {comparable_fixed['f1']:.3f}",
                f"- PR-AUC: {comparable_fixed['pr_auc']:.3f}",
                f"- ROC-AUC: {comparable_fixed['roc_auc']:.3f}",
                f"- Confusion matrix: TN={comparable_fixed['true_negatives']}, "
                f"FP={comparable_fixed['false_positives']}, "
                f"FN={comparable_fixed['false_negatives']}, "
                f"TP={comparable_fixed['true_positives']}",
            ]
        )
    if comparable_best:
        lines.extend(
            [
                "",
                "Best operating point on the same random split:",
                "",
                f"- Precision: {comparable_best['precision']:.3f}",
                f"- Recall: {comparable_best['recall']:.3f}",
                f"- F1: {comparable_best['f1']:.3f}",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_report(reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """Generate compact report files from model outputs."""
    metrics = _load_metrics(reports_dir)
    predictions_path = reports_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Model predictions not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    test_predictions = predictions[predictions["split"] == "test"].copy()

    _plot_confusion_matrix(test_predictions, reports_dir)
    _plot_curves(test_predictions, reports_dir)
    _write_error_slices(predictions, reports_dir)
    _plot_feature_importance(reports_dir)
    _write_summary(metrics, reports_dir)
    return metrics


if __name__ == "__main__":
    make_report()
    print(f"Report files written to {REPORTS_DIR}")

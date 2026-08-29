"""Run the complete limitation-detection workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

from lstm_model import REPORTS_DIR, run_model
from report_maker import make_report


def reset_reports_dir(reports_dir: Path = REPORTS_DIR) -> None:
    """Remove old generated outputs before a fresh run."""
    resolved = reports_dir.resolve()
    project_root = Path.cwd().resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe reports path: {resolved}")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    reports_dir.mkdir()


def main() -> None:
    """Train the model and generate reports."""
    reset_reports_dir()
    metrics = run_model(reports_dir=REPORTS_DIR)
    make_report(reports_dir=REPORTS_DIR)

    test = metrics["metrics"]["test"]
    print("Pipeline completed successfully.")
    print(f"Final dataset: {metrics['dataset']}")
    print(f"Model: {metrics['model']} ({metrics['feature_set']})")
    print(
        "Test metrics: "
        f"precision={test['precision']:.3f}, "
        f"recall={test['recall']:.3f}, "
        f"f1={test['f1']:.3f}, "
        f"pr_auc={test['pr_auc']:.3f}, "
        f"roc_auc={test['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()

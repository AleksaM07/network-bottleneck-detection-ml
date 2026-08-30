"""Run the complete limitation-detection workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

from lstm_model import REPORTS_DIR, run_model
from model_search import run_search
from report_maker import make_report


def reset_reports_dir(reports_dir: Path = REPORTS_DIR) -> None:
    """Remove generated outputs before a fresh run while preserving report images."""
    resolved = reports_dir.resolve()
    project_root = Path.cwd().resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe reports path: {resolved}")
    reports_dir.mkdir(exist_ok=True)
    for path in reports_dir.iterdir():
        if path.is_file() and path.name.startswith("report-v") and path.suffix == ".png":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    """Train the model and generate reports."""
    reset_reports_dir()
    # Augmentation is intentionally disabled by default. It stayed useful as an
    # experiment, but the 10k synthetic-limited run hurt chronological results.
    # from augment_limited_sequences import augment_limited_sequences
    # augment_limited_sequences(report_path=REPORTS_DIR / "augmentation_summary.csv")
    metrics = run_model(reports_dir=REPORTS_DIR)
    search_results = run_search(reports_dir=REPORTS_DIR)
    report_path = make_report(reports_dir=REPORTS_DIR)

    test = metrics["metrics"]["test"]
    best_practical = search_results[
        search_results["evaluation"] == "random_split_fixed_0_5"
    ].iloc[0]
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
    print(
        "Best notebook-comparable practical model: "
        f"{best_practical['model']} "
        f"f1={best_practical['f1']:.3f}, "
        f"precision={best_practical['precision']:.3f}, "
        f"recall={best_practical['recall']:.3f}"
    )
    print(f"Saved report image: {report_path}")


if __name__ == "__main__":
    main()

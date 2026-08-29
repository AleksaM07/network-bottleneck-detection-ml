# Legacy Scripts

These files are preserved for reference. They are not used by
`python orchestration.py`.

## Kept Files

- `latest_lstm_anomaly_time_series.ipynb`: latest historical LSTM notebook.
- `dataset_merger_original.py`: original HTTP download/upload dataset merge code.
- `unsupervised_anomaly_ensemble_original.py`: original unsupervised anomaly ensemble.
- `anomaly_visualizations_original.ipynb`: original anomaly visualization notebook.
- `random_forest_anomaly_original.py`: older random-forest anomaly attempt.
- `multiclass_random_forest_original.py`: older multiclass failure-category model.
- `data/`: anonymized datasets kept beside the legacy analysis files.

The latest LSTM notebook recorded:

- Accuracy: 0.9769
- Class 1 precision: 0.62
- Class 1 recall: 0.89
- Class 1 F1: 0.73
- Confusion matrix: `[[11983, 244], [49, 402]]`

That result is retained because it is useful project history. It is not the
current strict baseline because it used random splitting, identity-like fields,
SMOTE, and a one-timestep LSTM.

The notebook expects `merged_longer.xlsx`. That exact file was not found, so the
available anonymized data copies are stored in `data/`.

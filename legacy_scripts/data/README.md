# Legacy Data

These files are anonymized data copies kept beside the legacy notebooks/scripts.
Operator, route, mobility, and infrastructure labels are anonymized as stable
labels like `operator_01`, `route_001`, and so on.

- `merged_anonymized.csv`: anonymized version of the main active anomaly dataset.
- `mearged1_anonymized.csv`: anonymized version of the older smaller anomaly dataset.
- `data_final_deduplicated.csv`: same final deduplicated dataset as `../../data/data.csv`.

`latest_lstm_anomaly_time_series.ipynb` loads `merged_longer.xlsx`, but that exact
file is not present in the repository or in `old/`. These CSV files are the
closest available anonymized datasets for continuing that work without bringing
raw sensitive Excel exports back into the active project.

# Network Bottleneck Limitation Detection

This is a small reproducible ML project for detecting `Limitation` events in an
anonymized mobile/network measurement dataset.

`Limitation` is the binary target already present in the source export. The
available project files did not contain the original rule or manual process that
created it, so the model treats it as a provided label rather than a derived
field. Status columns such as transfer success do not fully explain the label.

## Project Files

```text
data/data.csv        Final anonymized, cleaned, de-duplicated dataset
legacy_scripts/      Renamed copies of useful old notebooks/scripts
lstm_model.py       Improved LSTM training and evaluation pipeline
report_maker.py     Compact report and plot generation
orchestration.py    One-command project runner
requirements.txt    Python dependencies
```

Generated outputs are written to `reports/`.

## Data

`data/data.csv` was created from the anonymized active anomaly dataset. The older
smaller `mearged1` dataset did not add unique rows after exact de-duplication.
Older multiclass/TCP workbooks were not merged into this file because they have
different schemas and different target definitions; mixing them into this binary
`Limitation` dataset would create invalid training rows.

The final dataset has:

- 43,551 rows
- 69 columns
- 1,697 positive `Limitation` rows
- dates from 2024-04-22 through 2024-05-25

Sensitive/context fields such as system IDs, client/server IPs, provisioning
traces, operator names, route names, mobility labels, and infrastructure labels
were anonymized before this final cleanup. Operators are stored as stable labels
like `operator_01`, routes as `route_001`, and so on. The repository no longer
keeps the original Excel exports or raw sensitive source files.

## Model

The final active model is an improved LSTM based on
`legacy_scripts/latest_lstm_anomaly_time_series.ipynb`.

The model uses the full anonymized CSV except `Limitation` and date/order
bookkeeping columns. Numeric throughput/radio/status fields are used directly
after scaling. Text/context fields such as operator, route, system, and server
pseudonyms are included through deterministic per-column hash features. The LSTM
builds real 5-row sequences within anonymized operator/system/server groups.

The split is chronological:

- Train: rows through 2024-05-10
- Validation: 2024-05-11 through 2024-05-15
- Test: rows after 2024-05-15

The classification threshold is selected on validation F1 only. The test split
is untouched until final evaluation.

Current chronological test result:

- Precision: 0.281
- Recall: 0.391
- F1: 0.327
- PR-AUC: 0.288
- ROC-AUC: 0.902

## Run

```bash
pip install -r requirements.txt
python orchestration.py
```

Main outputs:

- `reports/metrics.json`
- `reports/summary.md`
- `reports/predictions.csv`
- `reports/feature_importance.csv`
- `reports/legacy_comparable_metrics.json`
- `reports/legacy_comparable_training_history.csv`
- `reports/confusion_matrix.png`
- `reports/precision_recall_curve.png`
- `reports/roc_curve.png`
- `reports/errors_by_*.csv`

## Important Caveat

The historical notebook result below was much higher, but it used random
splitting, SMOTE, and a one-step LSTM. The active `lstm_model.py` keeps the LSTM
idea, uses the full anonymized CSV, and keeps chronological evaluation. It also
writes a notebook-comparable random-split result so the old score is not compared
against a harder future-date test by accident.

The latest historical notebook is preserved as
`legacy_scripts/latest_lstm_anomaly_time_series.ipynb`. Its recorded output was:

- Accuracy: 0.9769
- Class 1 precision: 0.62
- Class 1 recall: 0.89
- Class 1 F1: 0.73
- Confusion matrix: `[[11983, 244], [49, 402]]`

Current notebook-comparable random split result:

- Fixed threshold 0.5 F1: 0.718
- Best operating-point F1 on the same random split: 0.745
- PR-AUC: 0.809
- ROC-AUC: 0.991

Keep that result as historical context, not as the final scientific estimate.

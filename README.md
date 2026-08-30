# Network Bottleneck Limitation Detection

This is a small reproducible ML project for detecting `Limitation` events in an
anonymized mobile/network measurement dataset.

`Limitation` is the binary target already present in the source export. In this
project, positive labels represent Akamai edge/server-IP throughput-cap bugs:
repeated low/similar throughput on specific server IPs, with recovery after
changing server IP.

## Project Files

```text
data/data.csv        Final anonymized, cleaned, de-duplicated dataset
data/data_with_sequence.csv
                     Final dataset enriched with recovered Sequence/time fields
data/data_augmented.csv
                     Train-augmentation dataset with synthetic limited sequences
legacy_scripts/      Renamed copies of useful old notebooks/scripts
add_sequence_features.py
                     Recover Sequence/time fields from restored old HTTP DL files
augment_limited_sequences.py
                     Create conservative synthetic train-only limited sequences
lstm_model.py       Improved LSTM training and evaluation pipeline
model_search.py    Practical sklearn model benchmark on the same full CSV
report_maker.py     Single-image versioned report generation
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
keeps the original Excel exports or raw sensitive source files in the active
project tree.

`data/data_with_sequence.csv` is the current modeling dataset. It restores
`Sequence`, network-access time, anonymized location hash, sequence rank, and
sequence delta features from the restored legacy HTTP DL workbooks under `old/`.
The merge matched 43,357 of 43,551 rows and 1,695 of 1,697 positive rows.

`data/data_augmented.csv` is created from `data/data_with_sequence.csv`. It adds
10,000 synthetic limited rows generated only from real train-period positive
episodes with at least three consecutive limited sequences. Synthetic rows are
used only for training; validation/test rows remain real measurements.

## Model

The final active model is an improved LSTM based on
`legacy_scripts/latest_lstm_anomaly_time_series.ipynb`.

The model uses the full sequence-enriched anonymized CSV except `Limitation` and
date/order bookkeeping columns. Numeric throughput/radio/status/sequence fields
are used directly after scaling.
Text/context fields such as operator, route, system, and server pseudonyms are
included through deterministic per-column hash features. The LSTM builds real
5-row sequences within anonymized operator/system/server groups.

The feature engineering now includes the manual-analysis signal directly as ML
features: local sequence windows, previous/next throughput, server-change
context, same-day server throughput statistics, throughput/bandwidth ratios, and
past-only rolling history.

The split is chronological:

- Train: rows through 2024-05-10
- Validation: 2024-05-11 through 2024-05-15
- Test: rows after 2024-05-15

The classification threshold is selected on validation F1 only. The test split
is untouched until final evaluation.

Current chronological test result:

- Precision: 0.080
- Recall: 0.235
- F1: 0.120
- PR-AUC: 0.058
- ROC-AUC: 0.724

The model search currently finds that the strongest strict chronological
tabular candidate is `HistGradientBoostingClassifier`:

- Precision: 0.346
- Recall: 0.401
- F1: 0.372
- PR-AUC: 0.270
- ROC-AUC: 0.971

The most accurate notebook-comparable tabular candidate is
`LightGBM`:

- Fixed threshold 0.5 F1: 0.943
- PR-AUC: 0.988
- ROC-AUC: 0.999

## Run

```bash
pip install -r requirements.txt
python orchestration.py
```

The final output is a single versioned dashboard image:

- `reports/report-v1.png`
- `reports/report-v2.png`
- `reports/report-v3.png`
- ...

Intermediate CSV/JSON/model files are generated during the run and then removed.

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

- Best LightGBM fixed-threshold F1 on the same random split: 0.943

Keep that result as historical context, not as the final scientific estimate.

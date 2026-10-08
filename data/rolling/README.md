# Rolling SARIMAX checkpoint

The main hybrid forecasting workflow requires this matching pair:

- `oos_2023_rolling_causal_T1_a4dcc3a5c1cb.parquet`
- `oos_2023_rolling_causal_T1_a4dcc3a5c1cb.json`

The Parquet stores the 2023 out-of-sample SARIMAX predictions used to construct XGBoost residual targets. The JSON records the associated configuration, data signature, and completion metadata.

## Current package status

As checked on 8 October 2026, the required Parquet is absent from the repository tree. The available JSON describes an intermediate checkpoint with 336 completed hours, ending on 14 January 2023, rather than the complete 8,760-hour run.

Supply the original completed Parquet and its matching final JSON before executing the main workflow. Do not manually change completion flags to make an intermediate checkpoint appear complete.

## Expected completed checkpoint

The Parquet must contain:

- All 8,760 hourly target timestamps from 1 January to 31 December 2023.
- Prediction column `pred`.
- Refit-block identifier `block_num`.
- Forecast issuance timestamp `issue_time`.
- Model-fitting boundary `fit_end`.

The protocol uses 52 blocks of 168 hours and a final block of 24 hours. Each target is forecast one hour ahead. Fitting boundaries must precede the corresponding forecast blocks.

The JSON must identify the matching data and configuration and record a completed run, including `complete: true` and `completed_hours: 8760`.

## Why both files matter

The hybrid model is fitted using residuals derived from the saved out-of-sample predictions. Replacing those predictions with in-sample values or another rolling run changes the residual targets and may change the fitted correction.

The main script requires the complete checkpoint. It does not automatically recalculate missing or incomplete rolling predictions.

## Verification and execution

Run from the repository root:

```bash
python scripts/00_verify_reproducibility_inputs.py
python scripts/02_reproduce_models_and_figures.py
```

The verifier checks the Parquet checksum and temporal structure. The main script additionally checks the JSON completion status and its signature against the processed input and model configuration.

These checks establish file identity and temporal consistency within the rolling procedure. They do not independently validate every upstream screening or reconstruction decision.

This directory supports the main article workflow. The supplementary 48-hour LSTM input-window experiment is not included.

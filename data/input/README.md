# Processed hourly input dataset

`imputed_hourly_dataset.xlsx` is the processed hourly dataset supplied for the main forecasting workflow. It corresponds to the study workbook originally named `IMPUTAÇÃO A SER UTILIZADA.xlsx`.

The author has authorized sharing this processed dataset. Confidential raw operational measurements belonging to COPASA are not included.

## Dataset coverage

- Period: 1 January 2022 to 31 December 2024.
- Expected records: 26,304 hourly timestamps.
- Expected columns: 30.
- Timestamp column: `datetime`.
- Forecast target: `Vazão`, expressed in L/s.
- Development and fitting period: 2022–2023.
- Chronological evaluation period: 2024.

Preserve the workbook's column names, values, row order, and stored timestamps. Changes can invalidate its checksum or its compatibility with the rolling checkpoint.

## Use

From the repository root, run:

```bash
python scripts/00_verify_reproducibility_inputs.py
```

The verifier also requires the complete rolling Parquet file in `data/rolling`. Expected input checksums are recorded in `CHECKSUMS.study-inputs.sha256`.

Then run the main workflow:

```bash
python scripts/02_reproduce_models_and_figures.py
```

**The Excel workbook alone is insufficient for the main hybrid workflow.** The complete matching rolling Parquet and JSON are required to reconstruct the residual-learning stage.

## Scope

This workbook is the processed input used for forecasting. It is not a substitute for the confidential raw records needed to rerun reconstruction and the minimally processed inflow comparison from their original inputs.

The manuscript describes the screening, reconstruction, and subsequent hydrological consistency assessment.

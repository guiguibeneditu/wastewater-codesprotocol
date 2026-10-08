# External rolling checkpoint

The required checkpoint is `oos_2023_rolling_causal_T1_a4dcc3a5c1cb.parquet` with its JSON metadata. These data-derived artifacts are not distributed here pending authorization.

Expected fields are the target datetime index, prediction `pred`, refit `block_num`, `issue_time`, and `fit_end`. Once supplied, run `python scripts/00_verify_reproducibility_inputs.py` from the repository root. Timestamp alignment checks do not certify all upstream preprocessing decisions.

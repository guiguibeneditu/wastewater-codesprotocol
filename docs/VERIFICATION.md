# Package verification

Verification date: 2026-10-08 UTC.

- All four included Python scripts parsed successfully with Python's AST parser.
- Their bytes are unchanged from `wastewater-codesprotocol.zip` version 2.
- On the privately held source input, the dataset verifier passed: 26,304 rows, 30 columns, no missing values or duplicate timestamps, and a matching rolling configuration/data signature.
- The source Parquet SHA-256 matches the expected checksum.
- The full Parquet structural audit could not be rerun in the review environment because a Parquet engine was not installed there. `pyarrow` is already specified in the supplied requirements.
- No forecasting models were retrained and no performance results were independently regenerated during this packaging review.
- The standalone Q50 consistency-assessment implementation is absent from this archive; the Kalman script explicitly leaves it to a separate step.
- An obsolete README command pointing to `00_verify_rolling_checkpoint.py` was corrected to the existing `00_verify_reproducibility_inputs.py`.
- Restricted operational inputs and derived per-timestamp checkpoints are excluded. `.gitignore` no longer makes an exception that would track the treated dataset.

Before a submission release, reconcile the main manuscript, final outputs and Q50 implementation; identify the selected license and authorized data-access conditions; then record the repository commit or release used for the submission. No complete reproducibility certification is implied by these package checks.

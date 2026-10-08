# wastewater-codesprotocol

Code for hourly wastewater inflow forecasting with SARIMAX, a SARIMAX–XGBoost residual hybrid, and an LSTM-Q ensemble.

## Status and scope

This code package accompanies a manuscript being prepared for submission to Water Research. It is not a claim of journal acceptance or a completed end-to-end independent reproduction.

The four Python scripts are unchanged from the author's `wastewater-codesprotocol.zip` version 2. This publication package updates the documentation and excludes operational study data and data-derived checkpoints pending permission to distribute them. See `docs/VERIFICATION.md` for the checks performed and remaining limitations.

This repository covers the main article only. The supplementary latency experiment is outside the scope of this repository.

## Data access

Operational data originate from COPASA. This public package does **not** distribute the treated hourly dataset, the rolling prediction checkpoint, raw measurements, trained models, or per-timestamp outputs. Processing a dataset does not by itself establish permission to redistribute it. Any access request must be assessed by the author and the data owner under the applicable conditions; public access is not promised here.

Reproducing the study requires authorized access to the exact inputs below. Obtain them separately and place them at these local paths:

| Local path | Purpose |
|---|---|
| `data/input/imputed_hourly_dataset.xlsx` | Final treated hourly input for 2022–2024 |
| `data/rolling/oos_2023_rolling_causal_T1_a4dcc3a5c1cb.parquet` | 2023 rolling SARIMAX predictions used for XGBoost residual targets |
| `data/rolling/oos_2023_rolling_causal_T1_a4dcc3a5c1cb.json` | Associated checkpoint metadata |

The main input has 26,304 hourly timestamps. Development uses 2022–2023 and evaluation uses the 8,784 hours of 2024. The rolling checkpoint contains predictions for the 8,760 hours of 2023. File identities are documented in `CHECKSUMS.study-inputs.sha256`; the data are not included in this public package.

Exact hybrid reproduction requires the same rolling residual targets. Replacing them with in-sample predictions or a different numerical run may change the XGBoost fit.

## Workflow

| Script | Role |
|---|---|
| `scripts/00_verify_reproducibility_inputs.py` | Check authorized local study inputs and their signature |
| `scripts/01_kalman_reconstruction.py` | Kalman reconstruction and its validation |
| `scripts/02_reproduce_models_and_figures.py` | Main forecasting pipeline and figures |
| `scripts/04_flow_quality_ablation.py` | Comparison of minimally processed and treated development data |

The empirical Q50 consistency-assessment implementation is not included among these four scripts. Script 01 explicitly excludes that subsequent step. Full reconstruction from the operational record therefore requires the separate Q50 implementation and the relevant authorized inputs.

The chronological rolling procedure fits SARIMAX initially on 2022 and refits with an expanding window every 168 hours during 2023. Structural timestamp checks do not, by themselves, certify the information availability of every upstream reconstruction or screening decision. The manuscript must describe that scope accurately.

## Installation

Python 3.11 is recommended by the original package. Use an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

The Kalman environment uses `statsmodels==0.14.6` in `requirements-kalman.txt`; the forecasting environment specifies `statsmodels==0.14.4`. Use separate environments for these workflows. The scripts originated in Google Colab and retain runtime setup and path configuration blocks. Review those blocks for your environment before execution.

## Execution

After authorized inputs have been placed at the expected paths:

```bash
python scripts/00_verify_reproducibility_inputs.py
```

The verifier is expected to stop if those external files are absent. It requires the Parquet reader listed in `requirements.txt`.

For the main workflow, configure:

```bash
export SARIMAX_DATA_PATH="$PWD/data/input/imputed_hourly_dataset.xlsx"
export SARIMAX_CHECKPOINT_DIR="$PWD/data/rolling"
export SARIMAX_OUTPUT_DIR="$PWD/outputs/main_reproduction"
python scripts/02_reproduce_models_and_figures.py
```


Additional workflows require inputs that are not supplied:

| Workflow | Input or configuration |
|---|---|
| Kalman reconstruction | Workbook with missing flow; `KALMAN_INPUT_PATH` |
| Flow-quality ablation | Operational inflow input; `RAW_FLOW_PATH` |
| Flow-quality ablation | Treated predictions and metrics; `TREATED_PREDICTIONS_PATH` |

## Citation and license

Author: Guilherme Antônio Oliveira Benedito. ORCID: https://orcid.org/0009-0006-6636-7456


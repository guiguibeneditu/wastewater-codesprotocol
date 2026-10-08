#!/usr/bin/env python3
"""Verify the treated dataset and the causal 2023 rolling checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "rolling"
    / "oos_2023_rolling_causal_T1_a4dcc3a5c1cb.parquet"
)
DATASET_PATH = REPOSITORY_ROOT / "data" / "input" / "imputed_hourly_dataset.xlsx"
EXPECTED_PARQUET_SHA256 = "968867b712e6d1855760cb0bed97e7e8de4a42e27571fc3ce7b5e45affe3b91b"
EXPECTED_DATASET_SHA256 = "e41a2bd3a94b11c8c130311f7b11699e6304d53a9f1f18ea958035bcc297fc72"
EXPECTED_ROLLING_SIGNATURE = (
    "a4dcc3a5c1cb9ac5b8ee13d848a8890338267fdb1b8262001764ac19de82c3fb"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(path: Path = DATASET_PATH) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Treated dataset not found: {path}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_DATASET_SHA256:
        raise AssertionError(
            "Unexpected dataset checksum: "
            f"expected {EXPECTED_DATASET_SHA256}, obtained {actual_sha256}"
        )

    frame = pd.read_excel(path)
    required = {"datetime", "Vazão"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise AssertionError(f"Dataset is missing columns: {sorted(missing_columns)}")

    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if frame["datetime"].duplicated().any():
        raise AssertionError("The treated dataset contains duplicate timestamps.")
    frame = frame.sort_values("datetime").set_index("datetime")

    expected_index = pd.date_range(
        "2022-01-01 00:00:00", "2024-12-31 23:00:00", freq="h", name="datetime"
    )
    if not frame.index.equals(expected_index):
        raise AssertionError("The treated dataset is not the complete 2022-2024 hourly grid.")
    if frame.isna().any().any():
        raise AssertionError("The treated dataset contains missing values.")

    y = frame["Vazão"].copy()
    x_raw = frame.select_dtypes(include=[np.number]).drop(columns=["Vazão"]).copy()
    x_aligned = x_raw.shift(1)
    valid_index = x_aligned.index[y.notna() & x_aligned.notna().all(axis=1)]
    y = y.loc[valid_index]
    x_aligned = x_aligned.loc[valid_index]

    reference_end = pd.Timestamp("2023-12-31 23:00:00")
    y_reference = y.loc[:reference_end]
    x_reference = x_aligned.loc[y_reference.index]

    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(y_reference, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(x_reference, index=True).values.tobytes())
    digest.update("|".join(map(str, x_reference.columns)).encode("utf-8"))
    data_fingerprint = digest.hexdigest()

    signature_payload = {
        "pipeline_version": "rolling-causal-t1-r1-2026-08-05",
        "protocol": "rolling_expanding_refit_168h_one_step_observed_state_update",
        "target_index_semantics": "tau=t+1",
        "information_set": "X_tau_minus_1,Q_through_tau_minus_1",
        "data_start": "2022-01-01 00:00:00",
        "rolling_start": "2023-01-01 00:00:00",
        "rolling_end": "2023-12-31 23:00:00",
        "block_hours": 168,
        "sarimax_order": (1, 0, 0),
        "sarimax_seasonal_order": (1, 1, 1, 24),
        "exog_columns": list(map(str, x_aligned.columns)),
        "data_fingerprint": data_fingerprint,
        "statsmodels_version": "0.14.4",
    }
    serialized = json.dumps(
        signature_payload, ensure_ascii=False, sort_keys=True, default=str
    )
    rolling_signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if rolling_signature != EXPECTED_ROLLING_SIGNATURE:
        raise AssertionError(
            "The treated dataset does not reproduce the expected rolling signature: "
            f"expected {EXPECTED_ROLLING_SIGNATURE}, obtained {rolling_signature}"
        )

    return {
        "status": "PASS",
        "path": str(path),
        "sha256": actual_sha256,
        "rows": len(frame),
        "columns": len(frame.columns) + 1,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "rolling_signature": rolling_signature,
    }


def verify_rolling(path: Path = PARQUET_PATH) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Rolling checkpoint not found: {path}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_PARQUET_SHA256:
        raise AssertionError(
            "Unexpected Parquet checksum: "
            f"expected {EXPECTED_PARQUET_SHA256}, obtained {actual_sha256}"
        )

    frame = pd.read_parquet(path).sort_index()
    target = pd.DatetimeIndex(pd.to_datetime(frame.index), name="datetime")
    expected_target = pd.date_range(
        "2023-01-01 00:00:00", "2023-12-31 23:00:00", freq="h", name="datetime"
    )

    required = {"pred", "block_num", "issue_time", "fit_end"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns: {sorted(missing_columns)}")
    if not target.equals(expected_target):
        raise AssertionError("The target index is not the complete hourly grid for 2023.")
    if frame[list(required)].isna().any().any():
        raise AssertionError("The rolling checkpoint contains missing values.")
    if target.has_duplicates:
        raise AssertionError("The rolling checkpoint contains duplicate targets.")
    if not np.isfinite(frame["pred"].to_numpy(dtype=float)).all():
        raise AssertionError("The rolling checkpoint contains non-finite predictions.")

    issue = pd.DatetimeIndex(pd.to_datetime(frame["issue_time"]))
    if not issue.equals(target - pd.Timedelta(hours=1)):
        raise AssertionError("issue_time is not exactly one hour before every target.")

    expected_block = ((target - target[0]) // pd.Timedelta(hours=168) + 1).astype(int)
    observed_block = frame["block_num"].astype(int).to_numpy()
    if not np.array_equal(observed_block, expected_block):
        raise AssertionError("Rolling block assignments are inconsistent.")

    fit_end = pd.to_datetime(frame["fit_end"])
    block_sizes = frame.groupby("block_num", sort=True).size()
    if block_sizes.iloc[:-1].ne(168).any() or int(block_sizes.iloc[-1]) != 24:
        raise AssertionError("Expected 52 blocks of 168 h and one block of 24 h.")

    for block_number, block in frame.assign(_target=target).groupby("block_num"):
        block_fit_end = pd.to_datetime(block["fit_end"])
        if block_fit_end.nunique() != 1:
            raise AssertionError(f"Block {block_number} has multiple fit_end values.")
        if block_fit_end.iloc[0] != block["_target"].iloc[0] - pd.Timedelta(hours=1):
            raise AssertionError(f"Block {block_number} violates the causal fit boundary.")

    return {
        "status": "PASS",
        "path": str(path),
        "sha256": actual_sha256,
        "hours": len(frame),
        "target_start": str(target.min()),
        "target_end": str(target.max()),
        "blocks": int(block_sizes.size),
        "full_blocks_168h": int(block_sizes.eq(168).sum()),
        "final_block_hours": int(block_sizes.iloc[-1]),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_targets": int(target.duplicated().sum()),
        "causal_alignment_violations": 0,
    }


if __name__ == "__main__":
    report = {
        "status": "PASS",
        "treated_dataset": verify_dataset(),
        "rolling_checkpoint": verify_rolling(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

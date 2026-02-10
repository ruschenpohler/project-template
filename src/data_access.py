"""Governed data access layer for shadow datasets.

This module is the ONLY approved way to load data. It enforces:
  - Only files in data/shadow/ can be loaded.
  - The directory name must match the content_hash parameter.
  - (Phase 2) At load time, the actual SHA256 of the dataset files is
    recomputed and verified against the directory name. If they don't match,
    the load fails with a clear error.
  - (Phase 2) The dataset schema hash (column names, types) is computed and
    returned alongside the data.
  - Both hashes (content hash and schema hash) are logged to MLflow by
    run_experiment.py as Tier 1 parameters.

Do not modify this file without explicit human approval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SHADOW_ROOT = Path("data/shadow")

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ShadowDataset:
    """Container returned by load_shadow_dataset."""

    df: pd.DataFrame
    content_hash: str
    schema_hash: str


def compute_schema_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic hash of column names and dtypes."""
    schema_str = str([(col, str(dtype)) for col, dtype in zip(df.columns, df.dtypes)])
    return hashlib.sha256(schema_str.encode()).hexdigest()


# Phase 2 TODO: implement real CAS verification
# def _verify_content_hash(dataset_dir: Path, expected_hash: str) -> None:
#     """Recompute SHA256 of all files in dataset_dir and compare to
#     expected_hash (the directory name). Raise ValueError on mismatch."""
#     ...


def load_shadow_dataset(content_hash: str) -> ShadowDataset:
    """Load a dataset from data/shadow/{content_hash}/.

    In Phase 1 this returns a dummy DataFrame for end-to-end smoke testing.
    Phase 2 will add:
      - Real file loading from the content-addressed directory.
      - SHA256 verification: recompute the hash of the dataset files and
        reject the load if it doesn't match the directory name.
      - Schema hash computation returned alongside the data.

    Parameters
    ----------
    content_hash : str
        The SHA256 content hash that names the dataset directory.
        Must be a 64-character lowercase hex string.

    Returns
    -------
    ShadowDataset
        .df            -- the loaded DataFrame
        .content_hash  -- the verified content hash
        .schema_hash   -- SHA256 of (column_names, dtypes)

    Raises
    ------
    ValueError
        If content_hash is not a valid SHA256 hex string.
    FileNotFoundError
        If the dataset directory does not exist (Phase 2).
    """
    if not _SHA256_PATTERN.match(content_hash):
        raise ValueError(
            f"content_hash must be a 64-character lowercase hex string, "
            f"got '{content_hash}'"
        )

    dataset_dir = SHADOW_ROOT / content_hash

    # Phase 2 TODO: enforce that the directory exists
    # if not dataset_dir.is_dir():
    #     raise FileNotFoundError(
    #         f"No dataset directory found at {dataset_dir}. "
    #         f"Ensure the dataset has been placed in data/shadow/ with its "
    #         f"SHA256 hash as the directory name."
    #     )

    # Phase 2 TODO: load real data and verify CAS integrity
    # files = sorted(dataset_dir.glob("*.csv")) + sorted(dataset_dir.glob("*.parquet"))
    # _verify_content_hash(dataset_dir, content_hash)
    # df = pd.concat([pd.read_csv(f) for f in files])

    # Phase 1 stub: return a small dummy dataset for smoke testing
    df = pd.DataFrame(
        {
            "feature_1": [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.15, 0.6, 0.75, 0.5],
            "feature_2": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            "feature_3": [100, 200, 150, 300, 120, 280, 110, 250, 290, 180],
            "target": [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        }
    )

    schema_hash = compute_schema_hash(df)

    return ShadowDataset(
        df=df,
        content_hash=content_hash,
        schema_hash=schema_hash,
    )

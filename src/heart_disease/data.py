"""Loading and row-level cleaning for the UCI Heart Disease dataset.

Everything in this module works one row at a time, so it is safe to run on
the full dataset before the train/test split. Anything that learns from the
data (medians, modes, scaling) lives in ``pipeline.py`` and is fitted on the
training split only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "heart_disease"

# Columns that identify a row or its source rather than describe a patient.
ID_COLUMNS = ["id"]
LEAKY_OR_NON_FEATURE_COLUMNS = ["id", "dataset", "num"]

# A resting blood pressure or cholesterol of 0 is not physically possible.
# In this dataset a 0 means "not measured" (mostly the Switzerland subset).
ZERO_MEANS_MISSING = ["trestbps", "chol"]

BINARY_COLUMNS = ["sex", "fbs", "exang"]

_TRUE_VALUES = {True, "True", "TRUE", "true", 1, 1.0, "1"}
_FALSE_VALUES = {False, "False", "FALSE", "false", 0, 0.0, "0"}


def load_raw(path: str | Path = "heart_disease_uci.csv") -> pd.DataFrame:
    """Read the raw CSV exactly as published."""
    return pd.read_csv(path)


def _to_binary(series: pd.Series) -> pd.Series:
    """Map boolean-like values to 1.0 / 0.0 and keep missing values as NaN."""

    def convert(value):
        if pd.isna(value):
            return np.nan
        if value in _TRUE_VALUES:
            return 1.0
        if value in _FALSE_VALUES:
            return 0.0
        raise ValueError(f"Unexpected value {value!r} in column {series.name!r}")

    return series.map(convert).astype(float)


def drop_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that repeat the same patient record.

    The ``id`` column is unique for every row, so it has to be ignored when
    comparing rows, otherwise no duplicate can ever be found.
    """
    compare_cols = [c for c in df.columns if c not in ID_COLUMNS]
    return df.drop_duplicates(subset=compare_cols).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-level cleaning that does not learn anything from the data.

    * removes duplicate patient records (ignoring ``id``)
    * turns impossible zero values into missing values
    * encodes ``sex``, ``fbs`` and ``exang`` as 1.0 / 0.0, keeping NaN
    * builds the binary target from ``num`` (0 = no disease, 1-4 = disease)
    """
    df = drop_duplicate_patients(df.copy())

    for col in ZERO_MEANS_MISSING:
        df.loc[df[col] == 0, col] = np.nan

    df["sex"] = df["sex"].map({"Male": 1.0, "Female": 0.0})
    for col in ["fbs", "exang"]:
        df[col] = _to_binary(df[col])

    df[TARGET] = (df["num"] > 0).astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature table and the target, without id or source columns."""
    drop = [c for c in LEAKY_OR_NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop + [TARGET])
    y = df[TARGET]
    return X, y

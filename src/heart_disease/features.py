"""Clinical feature engineering.

These features use fixed medical thresholds, not statistics learned from the
data, so they can be computed before imputation. A missing input gives a
missing feature, which the pipeline then imputes using training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_ORDINAL = ["age_group", "chol_risk", "bp_category"]
ENGINEERED_NUMERIC = ["hr_reserve"]


def _bucket(series: pd.Series, bins: list[float]) -> pd.Series:
    """Cut a column into ordered buckets (0, 1, 2, ...), keeping NaN as NaN."""
    codes = pd.cut(series, bins=bins, labels=False, right=True)
    return codes.astype(float)


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add age group, cholesterol risk, blood pressure category and HR reserve.

    The original notebook used ``np.select(..., default=...)``, which silently
    put every missing cholesterol or blood pressure value into a real risk
    bucket. Here missing inputs stay missing.
    """
    X = X.copy()
    X["age_group"] = _bucket(X["age"], [-np.inf, 40, 55, 70, np.inf])
    X["chol_risk"] = _bucket(X["chol"], [-np.inf, 200, 240, np.inf])
    X["bp_category"] = _bucket(X["trestbps"], [-np.inf, 120, 140, np.inf])
    # Heart rate reserve: predicted max HR (220 - age) minus achieved max HR.
    X["hr_reserve"] = (220 - X["age"]) - X["thalch"]
    return X

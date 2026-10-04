"""A small synthetic table with the same schema and quirks as the UCI data.

The real CSV is not committed to the repo, so tests build their own data:
missing values in the same columns, zero cholesterol, boolean strings, and
a duplicated patient with a different id.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": rng.integers(29, 78, n),
            "sex": rng.choice(["Male", "Female"], n),
            "dataset": rng.choice(["Cleveland", "Hungary", "Switzerland", "VA Long Beach"], n),
            "cp": rng.choice(["typical angina", "atypical angina", "non-anginal", "asymptomatic"], n),
            "trestbps": rng.normal(132, 18, n).round(),
            "chol": rng.normal(240, 50, n).round(),
            "fbs": rng.choice([True, False], n),
            "restecg": rng.choice(["normal", "lv hypertrophy", "st-t abnormality"], n),
            "thalch": rng.normal(138, 25, n).round(),
            "exang": rng.choice([True, False], n),
            "oldpeak": rng.normal(0.9, 1.0, n).round(1),
            "slope": rng.choice(["upsloping", "flat", "downsloping"], n),
            "ca": rng.choice([0.0, 1.0, 2.0, 3.0], n),
            "thal": rng.choice(["normal", "fixed defect", "reversable defect"], n),
            "num": rng.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.15, 0.1, 0.1]),
        }
    )
    df["fbs"] = df["fbs"].astype(object)
    df["exang"] = df["exang"].astype(object)

    # Missingness patterns similar to the real data.
    for col, frac in {"ca": 0.6, "thal": 0.5, "slope": 0.3, "trestbps": 0.06, "thalch": 0.06, "fbs": 0.1}.items():
        df.loc[rng.random(n) < frac, col] = np.nan

    # Impossible zeros that mean "not measured".
    df.loc[[3, 7, 11], "chol"] = 0
    df.loc[[5], "trestbps"] = 0

    # The same patient recorded twice under different ids.
    dup = df.iloc[[10]].copy()
    dup["id"] = n + 1
    return pd.concat([df, dup], ignore_index=True)

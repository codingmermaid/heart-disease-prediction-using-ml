import numpy as np
import pandas as pd
import pytest

from heart_disease.data import TARGET, clean, drop_duplicate_patients, split_features_target


def test_duplicates_are_found_even_with_unique_ids(raw_df):
    # df.duplicated() on the full table returns 0 because id is always unique.
    assert raw_df.duplicated().sum() == 0
    assert len(drop_duplicate_patients(raw_df)) == len(raw_df) - 1


@pytest.mark.parametrize("column", ["chol", "trestbps", "fbs", "exang"])
def test_clean_deduplicates_normalized_patients(raw_df, column):
    patients = pd.concat([raw_df.iloc[[0]]] * 2, ignore_index=True)
    patients["id"] = [1, 2]
    patients[column] = [0, np.nan] if column in ["chol", "trestbps"] else [True, "TRUE"]
    original = patients.copy(deep=True)

    cleaned = clean(patients)

    assert len(cleaned) == 1
    assert cleaned["id"].tolist() == [1]
    if column in ["chol", "trestbps"]:
        assert pd.isna(cleaned.loc[0, column])
    else:
        assert cleaned.loc[0, column] == 1.0
    pd.testing.assert_frame_equal(patients, original)


def test_zero_blood_pressure_and_cholesterol_become_missing(raw_df):
    cleaned = clean(raw_df)
    assert (cleaned["chol"] == 0).sum() == 0
    assert (cleaned["trestbps"] == 0).sum() == 0
    assert cleaned.loc[3, "chol"] != cleaned.loc[3, "chol"]  # NaN


def test_binary_columns_keep_missing_values(raw_df):
    cleaned = clean(raw_df)
    assert set(cleaned["sex"].unique()) == {0.0, 1.0}
    assert cleaned["fbs"].isna().sum() == raw_df["fbs"].isna().sum()
    assert set(cleaned["fbs"].dropna().unique()) <= {0.0, 1.0}


@pytest.mark.parametrize("raw_value, expected", [("TRUE", 1.0), ("False", 0.0), (True, 1.0)])
def test_boolean_strings_are_parsed(raw_df, raw_value, expected):
    df = raw_df.copy()
    df.loc[0, "exang"] = raw_value
    assert clean(df).loc[0, "exang"] == expected


def test_target_is_binary(raw_df):
    cleaned = clean(raw_df)
    assert set(cleaned[TARGET].unique()) == {0, 1}
    assert (cleaned[TARGET] == (cleaned["num"] > 0)).all()


def test_features_exclude_id_source_and_raw_target(raw_df):
    X, y = split_features_target(clean(raw_df))
    for col in ["id", "dataset", "num", TARGET]:
        assert col not in X.columns
    assert len(X) == len(y)


def test_clean_does_not_fill_missing_values(raw_df):
    # Filling happens inside the pipeline, after the split.
    cleaned = clean(raw_df)
    assert cleaned["ca"].isna().sum() > 0
    assert cleaned["thal"].isna().sum() > 0

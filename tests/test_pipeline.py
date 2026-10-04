import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from heart_disease.data import clean, split_features_target
from heart_disease.features import add_clinical_features
from heart_disease.pipeline import MODEL_NAMES, NUMERIC_COLUMNS, build_model, build_preprocessor, feature_names
from heart_disease.train import cross_validate_model, evaluate


@pytest.fixture
def splits(raw_df):
    X, y = split_features_target(clean(raw_df))
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def test_imputer_learns_from_training_rows_only(splits):
    X_train, X_test, y_train, _ = splits
    # Make the test set very different so a leak would be obvious.
    X_test = X_test.copy()
    X_test["chol"] = 900.0

    pre = build_preprocessor().fit(X_train, y_train)
    numeric_imputer = pre.named_steps["columns"].named_transformers_["numeric"].named_steps["simpleimputer"]
    learned_chol_median = numeric_imputer.statistics_[NUMERIC_COLUMNS.index("chol")]

    assert learned_chol_median == pytest.approx(X_train["chol"].median())
    assert learned_chol_median != pytest.approx(pd.concat([X_train, X_test])["chol"].median())


def test_preprocessor_output_has_no_missing_values(splits):
    X_train, X_test, y_train, _ = splits
    pre = build_preprocessor().fit(X_train, y_train)
    assert not np.isnan(pre.transform(X_test)).any()


def test_missing_indicators_are_created(splits):
    X_train, _, y_train, _ = splits
    model = build_model("logreg").fit(X_train, y_train)
    names = feature_names(model)
    for col in ["ca", "thal", "slope"]:
        assert any("missingindicator" in n and n.endswith(col) for n in names), col


def test_unseen_category_does_not_crash(splits):
    X_train, X_test, y_train, _ = splits
    model = build_model("logreg").fit(X_train, y_train)
    X_new = X_test.head(3).copy()
    X_new["restecg"] = "category never seen in training"
    assert model.predict_proba(X_new).shape == (3, 2)


def test_engineered_features_stay_missing_when_inputs_are_missing():
    X = pd.DataFrame({"age": [50, 60], "chol": [np.nan, 250.0], "trestbps": [np.nan, 150.0], "thalch": [150.0, np.nan]})
    out = add_clinical_features(X)
    assert np.isnan(out.loc[0, "chol_risk"]) and out.loc[1, "chol_risk"] == 2
    assert np.isnan(out.loc[0, "bp_category"]) and out.loc[1, "bp_category"] == 2
    assert np.isnan(out.loc[1, "hr_reserve"])
    assert list(out["age_group"]) == [1, 2]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains_and_scores(splits, name):
    X_train, X_test, y_train, y_test = splits
    model = build_model(name).fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_cross_validation_runs_on_raw_features(splits):
    X_train, _, y_train, _ = splits
    # Raw features still contain NaN, which only works if preprocessing is in each fold.
    assert X_train.isna().any().any()
    scores = cross_validate_model("logreg", X_train, y_train, n_splits=3)
    assert set(scores) == {"accuracy", "f1", "roc_auc"}


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        build_model("not-a-model")

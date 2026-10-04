"""Leak-free preprocessing and model pipelines.

Imputation, missing-value indicators, one-hot encoding and scaling all learn
something from the data. Wrapping them in a scikit-learn ``Pipeline`` means
they are fitted on the training split only, and refitted inside every
cross-validation fold, so the test data never influences training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data import BINARY_COLUMNS
from .features import ENGINEERED_NUMERIC, ENGINEERED_ORDINAL, add_clinical_features

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"] + ENGINEERED_NUMERIC
CATEGORICAL_COLUMNS = ["cp", "restecg", "slope", "thal"]

RANDOM_STATE = 42


def _engineered_names(transformer, input_features):
    return list(input_features) + ENGINEERED_ORDINAL + ENGINEERED_NUMERIC


def build_preprocessor() -> Pipeline:
    """Feature engineering followed by per-column imputation and encoding."""
    numeric = make_pipeline(
        # add_indicator keeps the fact that a value was missing, which is
        # predictive here (ca is 66% missing, thal 53%, slope 34%).
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    ordinal = SimpleImputer(strategy="most_frequent")
    binary = SimpleImputer(strategy="most_frequent", add_indicator=True)
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    # Flags for missing categories are added separately, so they are not
    # one-hot encoded into redundant True/False column pairs.
    categorical_missing = MissingIndicator(features="missing-only")

    columns = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("ordinal", ordinal, ENGINEERED_ORDINAL),
            ("binary", binary, BINARY_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("categorical_missing", categorical_missing, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        [
            ("features", FunctionTransformer(add_clinical_features, feature_names_out=_engineered_names)),
            ("columns", columns),
        ]
    )


def _make_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")


MODEL_NAMES = ("logreg", "xgboost", "random_forest", "svm")


def build_model(name: str = "logreg") -> Pipeline:
    """Full pipeline: raw feature table in, class probabilities out."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", _make_classifier(name))])


def feature_names(model: Pipeline) -> list[str]:
    """Names of the columns the classifier actually sees, after encoding."""
    return list(model.named_steps["preprocess"].named_steps["columns"].get_feature_names_out())

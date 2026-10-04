"""Train and evaluate the models from the command line.

    python -m heart_disease.train --data heart_disease_uci.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import clean, load_raw, split_features_target
from .pipeline import MODEL_NAMES, RANDOM_STATE, build_model

CV_SCORING = ["accuracy", "f1", "roc_auc"]


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def cross_validate_model(name: str, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5):
    """Cross-validate the whole pipeline, so preprocessing is refitted per fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(build_model(name), X_train, y_train, cv=cv, scoring=CV_SCORING)
    return {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in CV_SCORING}


def run(data_path: str, models: list[str]) -> dict:
    X, y = split_features_target(clean(load_raw(data_path)))
    X_train, X_test, y_train, y_test = split(X, y)

    results = {}
    for name in models:
        model = build_model(name).fit(X_train, y_train)
        results[name] = {
            "test": evaluate(model, X_test, y_test),
            "cv": {m: {"mean": mean, "std": std} for m, (mean, std) in cross_validate_model(name, X_train, y_train).items()},
        }
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="heart_disease_uci.csv")
    parser.add_argument("--models", nargs="+", default=["logreg", "xgboost"], choices=MODEL_NAMES)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.models), indent=2))


if __name__ == "__main__":
    main()

"""Evaluate a small chronological multinomial baseline on generated data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .build_dataset import FEATURE_COLUMNS


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    truth = np.zeros_like(probabilities, dtype=float)
    for row_index, label in enumerate(y_true):
        truth[row_index, int(np.where(classes == label)[0][0])] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def evaluate(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path).sort_values(["date", "match_id"]).reset_index(drop=True)
    train = frame.loc[frame["split"] == "train"].copy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=7),
    )
    model.fit(train[FEATURE_COLUMNS].fillna(0.0), train["result"])
    classifier = model[-1]
    classes = classifier.classes_
    metrics: dict[str, object] = {
        "model": "standardized_multinomial_logistic_regression",
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(train)),
        "splits": {},
    }
    for split in ("validation", "test"):
        subset = frame.loc[frame["split"] == split].copy()
        if subset.empty:
            continue
        probabilities = model.predict_proba(subset[FEATURE_COLUMNS].fillna(0.0))
        prediction = classes[np.argmax(probabilities, axis=1)]
        metrics["splits"][split] = {
            "rows": int(len(subset)),
            "accuracy": float(accuracy_score(subset["result"], prediction)),
            "log_loss": float(log_loss(subset["result"], probabilities, labels=classes)),
            "multiclass_brier": multiclass_brier(subset["result"].to_numpy(), probabilities, classes),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/v0.1.0/pre_match_forecasting.parquet"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = evaluate(args.input)
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

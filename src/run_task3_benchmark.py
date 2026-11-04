
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/sample/flows.csv"
OUTPUT_PATH = ROOT / "data/raw/test_run.jsonl"

MODELS = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED)
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=SEED
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=SEED
    )
}

def main():
    df = pd.read_csv(DATA_PATH)

    possible_labels = [
        "label", "Label", "target", "Target",
        "class", "Class", "attack", "is_attack"
    ]

    target = next(
        (column for column in possible_labels if column in df.columns),
        None
    )

    if target is None:
        raise ValueError(
            f"Target column not found. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    if y.nunique() != 2:
        raise ValueError("This experiment expects binary classification.")

    if y.value_counts().min() < 2:
        raise ValueError(
            "Not enough samples in the minority class for stratified CV."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=2,
        n_repeats=10,
        random_state=SEED
    )

    batch_sizes = [1, 2, 4, 8, 16, 32]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            for model_name, factory in MODELS.items():

                model = factory()
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(
                    y_test, predictions, zero_division=0
                )
                recall = recall_score(
                    y_test, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_test, predictions, zero_division=0
                )

                tn, fp, fn, tp = confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ).ravel()

                fpr = fp / (fp + tn) if (fp + tn) else 0.0

                for batch_size in batch_sizes:

                    repeated_batch = pd.concat(
                        [X_test] * batch_size,
                        ignore_index=True
                    )

                    for repeat in range(25):

                        is_warmup = repeat < 5

                        start = time.perf_counter()

                        error = False

                        try:
                            model.predict(repeated_batch)
                        except Exception:
                            error = True

                        elapsed = time.perf_counter() - start

                        latency_ms = elapsed * 1000

                        throughput = (
                            batch_size * len(X_test) / elapsed
                            if elapsed > 0 and not error
                            else 0
                        )

                        record = {
                            "run_id": f"fold_{fold_id}_{model_name}_{batch_size}_{repeat}",
                            "fold": fold_id,
                            "model": model_name,
                            "phase": "warmup" if is_warmup else "evaluation",
                            "is_warmup": is_warmup,
                            "load_rps": batch_size,
                            "batch_size": batch_size,
                            "latency_ms": latency_ms,
                            "throughput_rps": throughput,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "fpr": fpr,
                            "error": error,
                            "status_code": 500 if error else 200
                        }

                        file.write(json.dumps(record) + "\n")

    print(f"Benchmark saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()


import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

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

from xgboost import XGBClassifier


# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/sample/flows.csv"
OUTPUT_PATH = ROOT / "data/raw/test_run.jsonl"

random.seed(SEED)
np.random.seed(SEED)


# ==================================================
# 2. MACHINE LEARNING MODELS
# ==================================================

MODELS = {
    "Logistic Regression": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )
    ),

    "Decision Tree": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            random_state=SEED
        )
    ),

    "Random Forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            random_state=SEED
        )
    ),

    "XGBoost": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1
        )
    )
}


# ==================================================
# 3. MAIN EXPERIMENT
# ==================================================

def main():

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    possible_labels = [
        "label",
        "Label",
        "target",
        "Target",
        "class",
        "Class",
        "attack",
        "is_attack"
    ]

    target = next(
        (
            column
            for column in possible_labels
            if column in df.columns
        ),
        None
    )

    if target is None:
        raise ValueError(
            "Target column not found. "
            f"Available columns: {list(df.columns)}"
        )

    if df[target].isna().any():
        raise ValueError(
            "The target column contains missing values."
        )

    # Separate input features and target
    X = df.drop(columns=[target])
    y = df[target]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No input features found.")

    # Encode target labels into 0 and 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Convert categorical features into numeric columns
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float
    )

    # Replace infinite values with NaN.
    # Missing values are imputed inside each model pipeline.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("No usable input features remain.")

    # Check binary classification
    unique_classes, class_counts = np.unique(
        y,
        return_counts=True
    )

    if len(unique_classes) != 2:
        raise ValueError(
            "This experiment expects binary classification."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Not enough samples in the minority class "
            "for stratified cross-validation."
        )

    print(f"Dataset shape: {X.shape}")
    print(f"Target column: {target}")
    print(f"Class distribution: {class_counts}")


    # ==================================================
    # 4. CROSS-VALIDATION
    # ==================================================

    cv = RepeatedStratifiedKFold(
        n_splits=2,
        n_repeats=10,
        random_state=SEED
    )

    batch_sizes = [1, 2, 4, 8, 16, 32]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_records = 0

    print("\nStarting benchmark...")


    # ==================================================
    # 5. BENCHMARK EXECUTION
    # ==================================================

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        for fold_id, (train_idx, test_idx) in enumerate(
            cv.split(X, y),
            start=1
        ):

            # X is a Pandas DataFrame
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            # y is a NumPy array
            y_train = y[train_idx]
            y_test = y[test_idx]

            for model_name, factory in MODELS.items():

                print(
                    f"Fold {fold_id}/20 | "
                    f"Model: {model_name}"
                )

                model = factory()

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(X_test)

                # Classification metrics
                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                tn, fp, fn, tp = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                ).ravel()

                fpr = (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else 0.0
                )


                # ==================================================
                # 6. COMPUTATIONAL PERFORMANCE MEASUREMENTS
                # ==================================================

                for batch_size in batch_sizes:

                    repeated_batch = pd.concat(
                        [X_test] * batch_size,
                        ignore_index=True
                    )

                    for repeat in range(25):

                        # First five measurements are warm-up
                        is_warmup = repeat < 5

                        start = time.perf_counter()

                        error = False

                        try:
                            model.predict(repeated_batch)

                        except Exception as exc:
                            error = True
                            print(
                                f"Prediction error: {exc}"
                            )

                        elapsed = (
                            time.perf_counter() - start
                        )

                        latency_ms = elapsed * 1000

                        sample_count = len(repeated_batch)

                        throughput = (
                            sample_count / elapsed
                            if elapsed > 0 and not error
                            else 0.0
                        )


                        # ==================================================
                        # 7. SAVE RAW MEASUREMENTS
                        # ==================================================

                        record = {
                            "run_id": (
                                f"fold_{fold_id}_"
                                f"{model_name}_"
                                f"{batch_size}_"
                                f"{repeat}"
                            ),

                            "fold": fold_id,
                            "model": model_name,

                            "phase": (
                                "warmup"
                                if is_warmup
                                else "evaluation"
                            ),

                            "is_warmup": is_warmup,

                            # This is the batch multiplier,
                            # not a measured requests-per-second value.
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

                            "status_code": (
                                500 if error else 200
                            )
                        }

                        file.write(
                            json.dumps(record) + "\n"
                        )

                        total_records += 1


    # ==================================================
    # 8. COMPLETION
    # ==================================================

    print("\nBENCHMARK COMPLETED")

    print(f"Total records: {total_records}")

    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

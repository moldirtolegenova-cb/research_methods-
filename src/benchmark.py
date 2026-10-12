
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path):
    """Load experiment settings from a YAML file."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")

    return config


def make_model(name, seed):
    """Create a classifier with reproducible settings."""
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )

    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=1,
        )

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )

    raise ValueError(f"Unknown model: {name}")


def evaluate(y_true, y_pred):
    """Calculate binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": fpr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark flow-based DDoS classifiers."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path)

    seed = config["experiment"]["random_seed"]
    data_path = ROOT / config["data"]["path"]
    target = config["data"]["target_column"]
    positive_label = config["data"]["positive_label"]

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    if data.empty:
        raise ValueError("The dataset is empty.")

    if target not in data.columns:
        raise ValueError(f"Missing target column: {target}")

    raw_target = data[target]

    if raw_target.isna().any():
        raise ValueError("Target labels contain missing values.")

    classes = set(raw_target.unique())

    if len(classes) != 2:
        raise ValueError(
            f"Expected exactly two target classes, found: {classes}"
        )

    if positive_label not in classes:
        raise ValueError(
            f"Configured positive_label {positive_label!r} "
            f"is not present in the target classes: {classes}"
        )

    # Encode the configured positive class as 1 and the other as 0.
    y = (raw_target == positive_label).astype(int)

    feature_sets = config["feature_sets"]
    models = config["models"]

    if not feature_sets:
        raise ValueError("At least one feature set must be configured.")

    if not models:
        raise ValueError("At least one model must be configured.")

    # Validate all configured features before splitting.
    all_features = sorted({
        feature
        for features in feature_sets.values()
        for feature in features
    })

    missing_features = sorted(set(all_features) - set(data.columns))
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    X_all = data[all_features].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X_all.to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN or infinite values.")

    # Use one identical stratified split for every configuration.
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config["evaluation"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    records = []

    for feature_set_name, features in feature_sets.items():
        if not features:
            raise ValueError(
                f"Feature set {feature_set_name!r} is empty."
            )

        X = data[features].apply(pd.to_numeric, errors="raise")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Feature set {feature_set_name!r} "
                "contains NaN or infinite values."
            )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        for model_name in models:
            model = make_model(model_name, seed)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            metrics = evaluate(y_test, predictions)

            is_baseline = (
                model_name == config["baseline"]["model"]
                and feature_set_name
                == config["baseline"]["feature_set"]
            )

            records.append({
                "model": model_name,
                "feature_set": feature_set_name,
                "n_features": len(features),
                "is_baseline": is_baseline,
                **metrics,
            })

    results = pd.DataFrame(records)

    baseline = results.loc[results["is_baseline"]]

    if len(baseline) != 1:
        raise ValueError(
            "Exactly one baseline result is required. "
            "Check the baseline model and feature set in config.yaml."
        )

    baseline_f1 = float(baseline.iloc[0]["f1"])
    practical_threshold = config["evaluation"][
        "practical_f1_difference"
    ]

    results["f1_difference_from_baseline"] = (
        results["f1"] - baseline_f1
    )

    results["practically_meaningful_difference"] = (
        results["f1_difference_from_baseline"].abs()
        >= practical_threshold
    )

    min_recall = config["guardrails"]["min_recall"]
    max_fpr = config["guardrails"]["max_false_positive_rate"]

    results["guardrails_pass"] = (
        (results["recall"] >= min_recall)
        & (results["false_positive_rate"] <= max_fpr)
    )

    output_path = ROOT / config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nBenchmark results:")
    print(results.to_string(index=False))
    print(f"\nBaseline F1-score: {baseline_f1:.4f}")
    print(f"Results saved to: {output_path.relative_to(ROOT)}")

    print(
        "\nNOTE: The included sample is synthetic and too small "
        "for research conclusions or statistical hypothesis testing."
    )

    summary = {
        "experiment": config["experiment"]["name"],
        "dataset_rows": len(data),
        "model_feature_configurations": len(results),
        "baseline_f1": baseline_f1,
        "positive_label": positive_label,
        "random_seed": seed,
        "output": str(output_path.relative_to(ROOT)),
    }

    print("\nRun summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

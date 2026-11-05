
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
BOOTSTRAP_ITERATIONS = 5000

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "test_run.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "metrics_summary.csv"


def to_bool(series):
    """Convert Boolean values and Boolean strings safely."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )



def load_raw_data():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw benchmark file not found: {RAW_PATH}"
        )

    records = []

    with RAW_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from error

    if not records:
        raise ValueError("The raw dataset is empty.")

    return pd.DataFrame(records)


def bootstrap_ci(
    values,
    statistic=np.median,
    iterations=BOOTSTRAP_ITERATIONS
):
    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(SEED)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True
    )

    estimates = statistic(samples, axis=1)

    lower = np.percentile(estimates, 2.5)
    upper = np.percentile(estimates, 97.5)

    return float(lower), float(upper)



def prepare_evaluation_data(df):

    required_columns = [
        "model",
        "phase",
        "is_warmup",
        "fold",
        "batch_size",
        "latency_ms",
        "throughput_rps",
        "f1_score",
        "fpr",
        "error"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["is_warmup"] = to_bool(df["is_warmup"])
    df["error"] = to_bool(df["error"])

    df["phase"] = (
        df["phase"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric_columns = [
        "fold",
        "batch_size",
        "latency_ms",
        "throughput_rps",
        "f1_score",
        "fpr"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    evaluation = df[
        (df["phase"] == "evaluation") &
        (~df["is_warmup"])
    ].copy()

    if evaluation.empty:
        raise ValueError(
            "No evaluation measurements remain after filtering."
        )

    return evaluation


def calculate_summary(evaluation):

    results = []

    grouped = evaluation.groupby(
        ["model", "batch_size"],
        sort=True
    )

    for (model, batch_size), group in grouped:

        successful = group[~group["error"]]

        latency = (
            successful["latency_ms"]
            .dropna()
            .to_numpy()
        )

        throughput = (
            successful["throughput_rps"]
            .dropna()
            .to_numpy()
        )


        if len(latency):

            median_latency = float(np.median(latency))

            p95_latency = float(
                np.percentile(latency, 95)
            )

            p99_latency = float(
                np.percentile(latency, 99)
            )

            latency_ci_low, latency_ci_high = bootstrap_ci(
                latency,
                statistic=np.median
            )

        else:

            median_latency = np.nan
            p95_latency = np.nan
            p99_latency = np.nan

            latency_ci_low = np.nan
            latency_ci_high = np.nan


        if len(throughput):

            median_throughput = float(
                np.median(throughput)
            )

            throughput_ci_low, throughput_ci_high = (
                bootstrap_ci(
                    throughput,
                    statistic=np.median
                )
            )

        else:

            median_throughput = np.nan
            throughput_ci_low = np.nan
            throughput_ci_high = np.nan


        fold_metrics = group.drop_duplicates(
            subset=["fold"]
        )

        f1_values = (
            fold_metrics["f1_score"]
            .dropna()
            .to_numpy()
        )

        fpr_values = (
            fold_metrics["fpr"]
            .dropna()
            .to_numpy()
        )

        if len(f1_values):

            mean_f1 = float(np.mean(f1_values))

            f1_ci_low, f1_ci_high = bootstrap_ci(
                f1_values,
                statistic=np.mean
            )

        else:

            mean_f1 = np.nan
            f1_ci_low = np.nan
            f1_ci_high = np.nan

        if len(fpr_values):

            mean_fpr = float(np.mean(fpr_values))

            fpr_ci_low, fpr_ci_high = bootstrap_ci(
                fpr_values,
                statistic=np.mean
            )

        else:

            mean_fpr = np.nan
            fpr_ci_low = np.nan
            fpr_ci_high = np.nan

        # Error rate guardrail

        total_runs = len(group)

        error_count = int(group["error"].sum())

        error_rate = (
            error_count / total_runs
            if total_runs > 0
            else np.nan
        )

        results.append({

            "model": model,

            "batch_size": int(batch_size),

            "n_evaluations": total_runs,

            "n_successful": len(successful),

            "median_latency_ms": median_latency,

            "p95_latency_ms": p95_latency,

            "p99_latency_ms": p99_latency,

            "latency_ci95_low": latency_ci_low,

            "latency_ci95_high": latency_ci_high,

            "median_throughput_samples_s": median_throughput,

            "throughput_ci95_low": throughput_ci_low,

            "throughput_ci95_high": throughput_ci_high,

            "mean_f1_score": mean_f1,

            "f1_ci95_low": f1_ci_low,

            "f1_ci95_high": f1_ci_high,

            "mean_fpr": mean_fpr,

            "fpr_ci95_low": fpr_ci_low,

            "fpr_ci95_high": fpr_ci_high,

            "error_count": error_count,

            "error_rate": error_rate
        })

    return pd.DataFrame(results)


def main():

    print("=" * 60)
    print("TASK 3: DATA PROCESSING AND STATISTICAL ANALYSIS")
    print("=" * 60)

    df = load_raw_data()

    print(f"\nRaw records: {len(df)}")

    warmup_mask = (
        (df["phase"].astype(str).str.lower() == "warmup") |
        to_bool(df["is_warmup"])
    )

    print(
        "Warm-up records:",
        int(warmup_mask.sum())
    )

    evaluation = prepare_evaluation_data(df)

    print(
        "Evaluation records:",
        len(evaluation)
    )

    summary = calculate_summary(evaluation)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.8f"
    )

    print("\nSTATISTICAL SUMMARY")

    print(
        summary.to_string(index=False)
    )

    print("\nSaved processed results to:")
    print(OUTPUT_PATH)

    print("\nProcessing completed successfully.")


if __name__ == "__main__":
    main()

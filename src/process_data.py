 
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BOOTSTRAP_ITERATIONS = 5000

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/test_run.jsonl"
OUTPUT_PATH = ROOT / "data/processed/metrics_summary.csv"

np.random.seed(SEED)


def bootstrap_ci(values, iterations=BOOTSTRAP_ITERATIONS,
                 statistic=np.median):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(SEED)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True
    )

    estimates = statistic(samples, axis=1)

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    )


def load_raw_data():
    records = []

    with RAW_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return pd.DataFrame(records)


def main():
    df = load_raw_data()

    required = [
        "model",
        "phase",
        "is_warmup",
        "latency_ms",
        "throughput_rps",
        "f1_score",
        "fpr",
        "error",
        "batch_size"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    evaluation = df[
        (df["phase"] == "evaluation") &
        (~df["is_warmup"].astype(bool))
    ].copy()

    evaluation["error"] = evaluation["error"].astype(bool)

    evaluation["latency_ms"] = pd.to_numeric(
        evaluation["latency_ms"], errors="coerce"
    )

    evaluation["throughput_rps"] = pd.to_numeric(
        evaluation["throughput_rps"], errors="coerce"
    )

    evaluation = evaluation.replace(
        [np.inf, -np.inf], np.nan
    )

    results = []

    for (model, batch_size), group in evaluation.groupby(
        ["model", "batch_size"]
    ):
        successful = group[~group["error"]]

        latency = successful["latency_ms"].dropna().to_numpy()
        throughput = successful["throughput_rps"].dropna().to_numpy()

        latency_low, latency_high = bootstrap_ci(latency)
        throughput_low, throughput_high = bootstrap_ci(throughput)

       
        fold_metrics = group.drop_duplicates(subset=["fold"])

        f1_values = fold_metrics["f1_score"].dropna().to_numpy()
        fpr_values = fold_metrics["fpr"].dropna().to_numpy()

        f1_low, f1_high = bootstrap_ci(
            f1_values, statistic=np.mean
        )
        fpr_low, fpr_high = bootstrap_ci(
            fpr_values, statistic=np.mean
        )

        total = len(group)
        errors = int(group["error"].sum())

        results.append({
            "model": model,
            "batch_size": int(batch_size),
            "n_evaluations": total,
            "median_latency_ms": (
                float(np.median(latency)) if len(latency) else np.nan
            ),
            "p95_latency_ms": (
                float(np.percentile(latency, 95)) if len(latency) else np.nan
            ),
            "p99_latency_ms": (
                float(np.percentile(latency, 99)) if len(latency) else np.nan
            ),
            "latency_ci95_low": latency_low,
            "latency_ci95_high": latency_high,
            "median_throughput_rps": (
                float(np.median(throughput))
                if len(throughput) else np.nan
            ),
            "throughput_ci95_low": throughput_low,
            "throughput_ci95_high": throughput_high,
            "mean_f1_score": (
                float(np.mean(f1_values)) if len(f1_values) else np.nan
            ),
            "f1_ci95_low": f1_low,
            "f1_ci95_high": f1_high,
            "mean_fpr": (
                float(np.mean(fpr_values)) if len(fpr_values) else np.nan
            ),
            "fpr_ci95_low": fpr_low,
            "fpr_ci95_high": fpr_high,
            "error_count": errors,
            "error_rate": errors / total if total else np.nan
        })

    summary = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nSTATISTICAL SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

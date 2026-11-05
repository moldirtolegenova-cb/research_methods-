
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "metrics_summary.csv"
)

RAW_PATH = (
    ROOT / "data" / "raw" / "test_run.jsonl"
)

FIGURES_DIR = (
    ROOT / "reports" / "figures"
)

SEED = 42

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind"
)

MARKERS = ["o", "s", "^", "D"]

LINESTYLES = ["-", "--", "-.", ":"]


def bootstrap_ci(values, iterations=5000):

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

    estimates = np.median(
        samples,
        axis=1
    )

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    )


def plot_latency(df):

    required = [
        "model",
        "batch_size",
        "median_latency_ms",
        "latency_ci95_low",
        "latency_ci95_high"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    models = sorted(
        df["model"].dropna().unique()
    )

    for index, model in enumerate(models):

        subset = (
            df[df["model"] == model]
            .sort_values("batch_size")
            .copy()
        )

        x = subset["batch_size"].to_numpy()

        y = subset["median_latency_ms"].to_numpy()

        low = subset["latency_ci95_low"].to_numpy()

        high = subset["latency_ci95_high"].to_numpy()

        ax.plot(
            x,
            y,
            label=model,
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[
                index % len(LINESTYLES)
            ],
            linewidth=2,
            markersize=6
        )

        valid = (
            np.isfinite(x) &
            np.isfinite(low) &
            np.isfinite(high)
        )

        ax.fill_between(
            x[valid],
            low[valid],
            high[valid],
            alpha=0.18
        )

    ax.set_title(
        "Inference Latency vs Batch Size"
    )

    ax.set_xlabel(
        "Batch multiplier (× test-fold size)"
    )

    ax.set_ylabel(
        "Inference latency (ms)"
    )

    ax.legend(
        title="Machine Learning Model"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        FIGURES_DIR / "fig1_latency_vs_load.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURES_DIR / "fig1_latency_vs_load.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Figure 1 exported successfully.")


def plot_distribution(raw):

    required = [
        "model",
        "phase",
        "is_warmup",
        "batch_size",
        "latency_ms",
        "error"
    ]

    missing = set(required) - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing raw columns: {sorted(missing)}"
        )

    raw = raw.copy()

    raw["is_warmup"] = (
        raw["is_warmup"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    raw["error"] = (
        raw["error"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    raw["phase"] = (
        raw["phase"]
        .astype(str)
        .str.lower()
    )

    raw["batch_size"] = pd.to_numeric(
        raw["batch_size"],
        errors="coerce"
    )

    raw["latency_ms"] = pd.to_numeric(
        raw["latency_ms"],
        errors="coerce"
    )

    raw = raw.replace(
        [np.inf, -np.inf],
        np.nan
    )

    evaluation = raw[
        (raw["phase"] == "evaluation") &
        (~raw["is_warmup"]) &
        (~raw["error"]) &
        (raw["batch_size"] == 1)
    ].copy()

    evaluation = evaluation.dropna(
        subset=["model", "latency_ms"]
    )

    if evaluation.empty:
        raise ValueError(
            "No valid evaluation data for boxplot."
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    order = sorted(
        evaluation["model"].unique()
    )

    sns.boxplot(
        data=evaluation,
        x="model",
        y="latency_ms",
        hue="model",
        order=order,
        hue_order=order,
        palette="colorblind",
        dodge=False,
        showfliers=True,
        legend=False,
        ax=ax
    )


    for position, model in enumerate(order):

        values = evaluation.loc[
            evaluation["model"] == model,
            "latency_ms"
        ].to_numpy()

        low, high = bootstrap_ci(values)

        median = np.median(values)

        ax.errorbar(
            position,
            median,
            yerr=[
                [median - low],
                [high - median]
            ],
            fmt="D",
            color="black",
            capsize=5,
            markersize=5,
            label=(
                "Median with 95% CI"
                if position == 0
                else None
            )
        )

    ax.set_title(
        "Inference Latency Distribution (Batch Multiplier = 1)"
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "Inference latency (ms)"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "fig2_distribution_boxplot.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURES_DIR / "fig2_distribution_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Figure 2 exported successfully.")


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed CSV not found. "
            "Run process_data.py first."
        )

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw JSONL file not found."
        )

    df = pd.read_csv(DATA_PATH)

    raw = pd.read_json(
        RAW_PATH,
        lines=True
    )

    plot_latency(df)

    plot_distribution(raw)

    print("\nAll scientific figures generated.")

    print(
        "Output directory:",
        FIGURES_DIR
    )


if __name__ == "__main__":
    main()

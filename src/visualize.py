
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/metrics_summary.csv"
RAW_PATH = ROOT / "data/raw/test_run.jsonl"
FIGURES_DIR = ROOT / "reports/figures"

sns.set_theme(
    style="whitegrid",
    context="paper"
)

MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]


# ==================================================
# 2. LATENCY VS BATCH LOAD
# ==================================================

def plot_latency(df):

    required_columns = [
        "model",
        "batch_size",
        "median_latency_ms",
        "latency_ci95_low",
        "latency_ci95_high"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in metrics_summary.csv: {missing}"
        )

    df = df.copy()

    numeric_columns = [
        "batch_size",
        "median_latency_ms",
        "latency_ci95_low",
        "latency_ci95_high"
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

    df = df.dropna(
        subset=["model", "batch_size", "median_latency_ms"]
    )

    if df.empty:
        raise ValueError(
            "No valid latency data available for plotting."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    models = df["model"].unique()

    for index, model in enumerate(models):

        subset = (
            df[df["model"] == model]
            .sort_values("batch_size")
        )

        x = subset["batch_size"].to_numpy()
        y = subset["median_latency_ms"].to_numpy()

        marker = MARKERS[index % len(MARKERS)]
        linestyle = LINESTYLES[index % len(LINESTYLES)]

        ax.plot(
            x,
            y,
            label=model,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8
        )

        # Bootstrap 95% confidence interval
        ci_data = subset.dropna(
            subset=[
                "latency_ci95_low",
                "latency_ci95_high"
            ]
        )

        if not ci_data.empty:

            ax.fill_between(
                ci_data["batch_size"].to_numpy(),
                ci_data["latency_ci95_low"].to_numpy(),
                ci_data["latency_ci95_high"].to_numpy(),
                alpha=0.15
            )

    ax.set_title("Inference Latency vs Batch Load")
    ax.set_xlabel("Batch multiplier")
    ax.set_ylabel("Inference latency (ms)")

    ax.legend(title="Classifier")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    fig.tight_layout()

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

    print("Figure 1 generated successfully.")


# ==================================================
# 3. LATENCY DISTRIBUTION
# ==================================================

def plot_distribution(raw):

    required_columns = [
        "model",
        "phase",
        "is_warmup",
        "latency_ms",
        "error"
    ]

    missing = set(required_columns) - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing columns in raw data: {missing}"
        )

    raw = raw.copy()

    # Normalize Boolean columns
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

    raw["latency_ms"] = pd.to_numeric(
        raw["latency_ms"],
        errors="coerce"
    )

    raw = raw.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep only successful evaluation measurements
    raw = raw[
        (raw["phase"] == "evaluation") &
        (~raw["is_warmup"]) &
        (~raw["error"])
    ].copy()

    raw = raw.dropna(
        subset=["model", "latency_ms"]
    )

    if raw.empty:
        raise ValueError(
            "No valid evaluation measurements available "
            "for the latency distribution plot."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=raw,
        x="model",
        y="latency_ms",
        hue="model",
        palette="colorblind",
        legend=False,
        ax=ax
    )

    ax.set_title("Distribution of Inference Latency")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Inference latency (ms)")

    ax.tick_params(
        axis="x",
        rotation=20
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

    print("Figure 2 generated successfully.")


# ==================================================
# 4. MAIN
# ==================================================

def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}. "
            "Run process_data.py first."
        )

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_PATH}. "
            "Run run_task3_benchmark.py first."
        )

    print("Loading processed data...")

    df = pd.read_csv(DATA_PATH)

    print("Loading raw benchmark measurements...")

    raw = pd.read_json(
        RAW_PATH,
        lines=True
    )

    print("\nGenerating scientific visualizations...")

    plot_latency(df)

    plot_distribution(raw)

    print("\nBoth figures exported successfully.")

    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

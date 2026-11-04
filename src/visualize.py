
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/metrics_summary.csv"
FIGURES_DIR = ROOT / "reports/figures"

sns.set_theme(style="whitegrid", context="paper")

PALETTE = sns.color_palette("colorblind")

MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]


def plot_latency(df):
    fig, ax = plt.subplots(figsize=(9, 6))

    models = df["model"].unique()

    for index, model in enumerate(models):
        subset = df[df["model"] == model].sort_values("batch_size")

        x = subset["batch_size"].to_numpy()
        y = subset["median_latency_ms"].to_numpy()
        low = subset["latency_ci95_low"].to_numpy()
        high = subset["latency_ci95_high"].to_numpy()

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

        ax.fill_between(
            x,
            low,
            high,
            alpha=0.15
        )

    ax.set_title("Inference Latency vs Batch Load")
    ax.set_xlabel("Batch size (samples)")
    ax.set_ylabel("Inference latency (ms)")
    ax.legend(title="Classifier")
    ax.grid(True, linestyle="--", alpha=0.5)

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


def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="model",
        y="median_latency_ms",
        hue="model",
        palette="colorblind",
        legend=False,
        ax=ax
    )

    ax.set_title("Distribution of Median Inference Latency")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Median inference latency (ms)")
    ax.tick_params(axis="x", rotation=20)

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


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    plot_latency(df)
    plot_distribution(df)

    print("Both figures exported successfully.")
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

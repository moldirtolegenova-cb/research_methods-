# Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection

## Authors

- Kudaibergen Nazerke
- Tolegenova Moldir

## Overview

This project investigates how machine learning classifier choice and feature-set selection affect flow-based Distributed Denial of Service (DDoS) attack detection.

The benchmark compares four classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Each classifier is evaluated using two predefined feature sets: `full` and `selected`. The pipeline calculates classification metrics, compares each configuration with a baseline, checks predefined performance guardrails, and saves the results as a CSV file.

**Important:** The included dataset contains only ten synthetic records. It is intended to verify that the pipeline runs correctly. It is not representative of real network traffic and cannot support conclusions about real-world DDoS detection.

## Research Question

How do different machine learning classifiers and feature selection strategies affect the performance of flow-based DDoS attack detection?

## Research Objective

To establish a reproducible experimental pipeline for comparing four machine learning classifiers and two predefined feature sets under consistent evaluation conditions.

## Hypotheses

**H0 — Null hypothesis:** Under the same dataset, preprocessing, train-test split, and evaluation protocol, the evaluated classifiers and feature sets do not produce a practically meaningful difference in F1-score.

**H1 — Alternative hypothesis:** At least one evaluated classifier or feature set produces a practically meaningful difference in F1-score.

A difference of 0.02 in F1-score is defined as the minimum practically meaningful difference for the planned comparison. The current synthetic dataset is insufficient to test these hypotheses reliably.

## Evaluation Metrics

**Primary metric:** F1-score.

**Secondary metrics:**

- Precision
- Recall
- Accuracy
- False-positive rate (FPR)

The predefined performance targets are:

- Recall ≥ 0.90
- FPR ≤ 0.05

These are targets, not guaranteed results.

## Baseline

The baseline is Logistic Regression trained using the full feature set.

Other configurations are compared with the baseline using their F1-score difference.

## Repository Structure

```text
research_methods-/
├── configs/
│   └── config.yaml
├── data/
│   └── sample/
│       └── flows.csv
├── docs/
│   └── methodology.md
├── notebooks/
├── results/
│   └── .gitkeep
├── src/
│   └── benchmark.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `results/benchmark_results.csv` file is generated when the benchmark runs. It may be included in the repository as an example output.

## System Requirements

- Python 3.13 (tested environment)
- pip
- Internet access for installing dependencies

## Quickstart Guide

### 1. Clone the repository

```bash
git clone https://github.com/moldirtolegenova-cb/research_methods-.git
cd research_methods-
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the benchmark

```bash
python src/benchmark.py --config configs/config.yaml
```

The script prints the evaluation results and saves them to:

```text
results/benchmark_results.csv
```

## Experimental Protocol

1. Load the dataset from the configured path.
2. Validate the target labels and input features.
3. Convert the configured positive label to the positive binary class.
4. Create one stratified train-test split using the configured random seed.
5. Train each classifier using both feature sets.
6. Calculate accuracy, precision, recall, F1-score, and FPR.
7. Compare each configuration with the baseline.
8. Check whether the recall and FPR guardrails are satisfied.
9. Save the results as a CSV file.

All configurations use the same train-test partition to make the comparisons consistent.

## Data

The file `data/sample/flows.csv` contains ten synthetic records with five flow-based features and a binary label.

The features are:

- `flow_duration`
- `total_fwd_packets`
- `total_bwd_packets`
- `flow_bytes_per_sec`
- `packet_rate`

The target column is `label`, where `0` represents benign traffic and `1` represents DDoS traffic in the included sample.

For a full research experiment, replace or supplement the synthetic sample with a suitable labeled flow-based network traffic dataset. Document the source, license, label mapping, preprocessing decisions, and any exclusions.

## Limitations

- The synthetic sample is extremely small.
- The synthetic records are not representative of real network traffic.
- The single train-test split does not provide a robust estimate of model performance.
- No statistical significance can be established from the current sample.
- Results should not be interpreted as evidence that one classifier is superior for real-world DDoS detection.

A full experiment should use representative data, suitable preprocessing, repeated evaluation or cross-validation, and an appropriate statistical comparison.

## Reproducibility

The repository includes pinned Python dependencies, an experiment configuration, a runnable benchmark script, and a sample dataset.

The random seed, feature sets, models, baseline, evaluation settings, and guardrails are defined in `configs/config.yaml`.

## Citation

Kudaibergen Nazerke and Tolegenova Moldir. (2026). *Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection*. GitHub repository.

## License

This project is licensed under the MIT License. See `LICENSE`.
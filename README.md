
# Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection

## Authors
- [Kudaibergen Nazerke]
- [Tolegenova Moldir]

## Overview
This project investigates how classifier selection and feature selection affect
flow-based Distributed Denial of Service (DDoS) attack detection.

The benchmark compares Logistic Regression, Decision Tree, Random Forest,
and XGBoost using a full feature set and a predefined selected feature subset.

The included synthetic sample verifies pipeline execution only. It is not
representative of real-world network traffic and cannot support research conclusions.

## Research Question
How do different machine learning classifiers and feature selection strategies
affect the performance of flow-based DDoS attack detection?

## Hypotheses
H0: The evaluated classifiers and feature sets do not produce a practically
meaningful difference in F1-score under identical evaluation conditions.

H1: At least one classifier or feature set produces a practically meaningful
difference in F1-score under identical evaluation conditions.

## Primary Metric
F1-score.

## Secondary and Guardrail Metrics
- Precision
- Recall (target: at least 0.90)
- Accuracy
- False-positive rate (target: at most 0.05)

The thresholds are evaluation targets, not guaranteed results.

## Baseline
Logistic Regression trained on the full feature set.

## Repository Structure
```text
ddos-ml-benchmark/
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

## System Requirements
- Python 3.11
- pip
- Internet access for installing dependencies

## Quickstart Guide

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install pinned dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the benchmark

```bash
python src/benchmark.py --config configs/config.yaml
```

The script prints the evaluation metrics and writes the results to
`results/benchmark_results.csv`.

## Configuration
Experiment parameters are stored in `configs/config.yaml`.
The configuration defines the dataset path, target label, feature sets,
classifier list, baseline, evaluation settings, and guardrail thresholds.

## Experimental Protocol
All configurations use the same train-test partition and random seed.
The same target labels and evaluation metrics are applied to each model.
The selected feature subset is defined before evaluation.

## Planned Benchmarks

| Model | Full Features | Selected Features |
|---|---|---|
| Logistic Regression (baseline) | Yes | Yes |
| Decision Tree | Yes | Yes |
| Random Forest | Yes | Yes |
| XGBoost | Yes | Yes |

## Data
`data/sample/flows.csv` contains ten synthetic records for pipeline testing.
Replace or supplement it with a suitable labeled flow-based network traffic
dataset for the actual research experiment. Document the dataset source,
license, label mapping, and preprocessing decisions.

## Limitations
A small synthetic sample is not sufficient to estimate real-world detection
performance, establish statistical significance, or validate the hypotheses.
The final experiment requires representative data and an appropriate
evaluation protocol.

## Citation
If this repository is used in academic work, cite the project as:

[Author(s)]. (2026). Comparative Evaluation of Machine Learning Models
for Flow-Based DDoS Attack Detection. GitHub repository.

## License
This project is licensed under the MIT License. See `LICENSE`.
# Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection

## 1. Research Overview

This research evaluates four machine learning classifiers
for detecting Distributed Denial of Service (DDoS) attacks
using network flow-based features.

The study compares classification quality and
computational performance under different batch sizes.

The evaluated models are:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The main objective is to investigate the trade-off
between detection quality and inference efficiency.

**Important limitation:** The initial classification
dataset contains only ten synthetic observations.
The results are preliminary and cannot be generalized
to real-world network traffic.

## 2. Authors

- Kudaibergen Nazerke
- Tolegenova Moldir

**Course:** Research Methods

**Year:** 2026

## 3. Research Questions

1. Which classifier provides the best F1-score?
2. Which classifier has the lowest false positive rate?
3. How does batch size affect inference latency?
4. Which classifier provides the best computational efficiency?
5. How reliable are the benchmark measurements?

## 4. Research Objective and Hypotheses

### Research Objective

To compare four machine learning classifiers
using classification metrics and computational
performance measurements.

### Null Hypothesis (H0)

There is no practically meaningful difference
in F1-score between the evaluated classifiers.

### Alternative Hypothesis (H1)

At least one evaluated classifier produces
a practically meaningful difference in F1-score.

A difference of 0.02 in F1-score is defined
as the minimum practically meaningful difference.

The small synthetic dataset does not provide
sufficient evidence for a reliable statistical
test of these hypotheses.

## 5. Dataset and Input Features

The classification dataset is stored at:

`data/sample/flows.csv`

The target column is:

`label`

Class definitions:

- 0 — Benign traffic
- 1 — DDoS traffic

The input features are:

| Feature | Description |
|---|---|
| flow_duration | Duration of the network flow |
| total_fwd_packets | Total forward packets |
| total_bwd_packets | Total backward packets |
| flow_bytes_per_sec | Flow bytes per second |
| packet_rate | Packet rate |

The current sample contains ten synthetic records:
five benign and five DDoS observations.

It is used to test the experimental pipeline,
not to establish production-level detection accuracy.

## 6. Machine Learning Models

The following classifiers are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The baseline classifier is Logistic Regression.

The alternative classifiers are compared
with the baseline.

The current Task 3 benchmark uses the full
input feature set.

## 7. Evaluation Metrics

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)

### Computational Metrics

- Median inference latency (ms)
- p95 latency (ms)
- p99 latency (ms)
- Throughput (samples/s)
- Error rate

### Statistical Methods

The analysis uses:

- Median
- 95th percentile
- 99th percentile
- Bootstrap 95% confidence intervals
- 5000 bootstrap iterations
- Random seed 42

Warm-up measurements are excluded
from the final statistical analysis.

## 8. Experimental Setup

The benchmark uses repeated stratified
cross-validation.

The computational experiment includes:

- Four machine learning models.
- Six batch multipliers: 1, 2, 4, 8, 16 and 32.
- Twenty-five timing repetitions per fold and batch.
- Five warm-up repetitions.
- Twenty evaluation repetitions.

The raw benchmark file contains 12,000 records:

- 2,400 warm-up records.
- 9,600 evaluation records.

The batch multiplier represents a multiple
of the test-fold size.

Throughput is measured as processed samples
per elapsed second.

The benchmark uses a fixed random seed of 42.

## 9. Results and Discussion

### 9.1 Baseline vs Proposed Method

The baseline is Logistic Regression.

For the current preliminary run, Decision Tree
is selected as the alternative configuration
because it achieved the same observed classification
metrics with lower measured inference latency.

The following table summarizes the existing
benchmark run. Latency and throughput are aggregated
across the six batch multipliers.

| Metric | Baseline: Logistic Regression | Proposed: Decision Tree |
|---|---:|---:|
| F1-score | 1.000 | 1.000 |
| False Positive Rate | 0.000 | 0.000 |
| Median Latency (ms) | 2.128 | 1.763 |
| p95 Latency (ms) | 3.399 | 2.826 |
| p99 Latency (ms) | 3.714 | 3.111 |
| Throughput (samples/s) | 10474.31 | 12375.85 |
| Error Rate | 0.000 | 0.000 |

**Interpretation:**

In this preliminary benchmark run, Decision Tree
showed lower median latency and higher median
throughput than Logistic Regression.

Both models produced the same observed F1-score
and false positive rate.

These findings are limited to the small synthetic
classification dataset and the recorded hardware
environment.

They do not establish that Decision Tree
is generally superior for real-world DDoS detection.

The table should be regenerated if the benchmark
is executed again on a different environment.

### 9.2 Statistical Processing

Processed results are saved in:

`data/processed/metrics_summary.csv`

The CSV includes:

- Number of evaluation measurements.
- Number of successful measurements.
- Median latency.
- p95 latency.
- p99 latency.
- Bootstrap 95% confidence intervals.
- Median throughput.
- F1-score.
- False positive rate.
- Error count.
- Error rate.

### 9.3 Scientific Visualization

#### Figure 1 — Inference Latency vs Batch Size

The first figure compares inference latency
across the four machine learning models.

The shaded areas represent bootstrap
95% confidence intervals.

![Latency vs Batch Size](reports/figures/fig1_latency_vs_load.png)

[Download vector PDF figure](reports/figures/fig1_latency_vs_load.pdf)

#### Figure 2 — Latency Distribution

The second figure shows the distribution
of successful evaluation latency measurements
at batch multiplier 1.

The boxplots display the distribution,
while the error bars indicate bootstrap
95% confidence intervals of the median.

![Latency Distribution](reports/figures/fig2_distribution_boxplot.png)

[Download vector PDF figure](reports/figures/fig2_distribution_boxplot.pdf)

## 10. Domain Model

The formal mathematical model includes:

- Dataset definition.
- Binary classification model.
- Binary Cross-Entropy Loss.
- Classification metrics.
- Computational metrics.
- Statistical notation.
- Bootstrap confidence intervals.

See:

[Domain Model](docs/domain_model.md)

## 11. Repository Structure

```text
research_methods-/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── test_run.jsonl
│   ├── processed/
│   │   └── metrics_summary.csv
│   └── sample/
│       └── flows.csv
├── docs/
│   ├── domain_model.md
│   └── methodology.md
├── notebooks/
│   └── 01_data_analysis.ipynb
├── reports/
│   └── figures/
│       ├── fig1_latency_vs_load.pdf
│       ├── fig1_latency_vs_load.png
│       ├── fig2_distribution_boxplot.pdf
│       └── fig2_distribution_boxplot.png
├── results/
│   └── benchmark_results.csv
├── src/
│   ├── benchmark.py
│   ├── run_task3_benchmark.py
│   ├── process_data.py
│   └── visualize.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 12. Installation

### Requirements

- Python 3.11 or newer.
- pip.
- Git.

### Clone Repository

```bash
git clone https://github.com/moldirtolegenova-cb/research_methods-.git

cd research_methods-
```

### Create Virtual Environment

Windows:

```powershell
py -m venv .venv

.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 13. Reproducibility Guide

### Step 1. Generate Benchmark Measurements

```bash
python src/run_task3_benchmark.py
```

This generates:

`data/raw/test_run.jsonl`

### Step 2. Process Raw Measurements

```bash
python src/process_data.py
```

This generates:

`data/processed/metrics_summary.csv`

### Step 3. Generate Scientific Figures

```bash
python src/visualize.py
```

This generates four figure files under:

`reports/figures/`

### Step 4. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

Run all cells sequentially from the beginning.

The notebook displays the data analysis,
statistical summaries and visualizations.

## 14. Performance Guardrails

The predefined classification targets are:

- Recall >= 0.90
- False Positive Rate <= 0.05

The error rate is monitored as an additional
computational reliability metric.

These thresholds are evaluation targets,
not guaranteed experimental outcomes.

## 15. Limitations

The research has several limitations:

1. The initial dataset contains only ten synthetic records.
2. The dataset does not represent real network traffic.
3. Repeated cross-validation folds are not fully independent.
4. Hardware differences may affect inference latency.
5. A larger representative dataset is necessary
   to validate detection performance.
6. The current experiment does not establish
   production-level DDoS detection accuracy.

Future research should use a larger labeled
network traffic dataset and more extensive
experimental validation.

## 16. Conclusion

This project implements a reproducible experimental
pipeline for comparing machine learning classifiers
for flow-based DDoS attack detection.

The pipeline includes raw benchmark collection,
warm-up filtering, statistical aggregation,
confidence interval estimation and scientific
visualization.

The current results demonstrate the operation
of the experimental workflow.

Further research with representative network
traffic data is required before drawing
general conclusions about real-world detection.

## 17. Citation

Kudaibergen Nazerke and Tolegenova Moldir (2026).

*Comparative Evaluation of Machine Learning Models
for Flow-Based DDoS Attack Detection.*

Research Methods project.

## 18. License

This project is licensed under the MIT License.
See the LICENSE file for details.
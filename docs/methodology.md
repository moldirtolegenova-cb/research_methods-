# Methodology Passport Card

## 1. Project Information

**Title:** Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection

**Authors:** Kudaibergen Nazerke and Tolegenova Moldir

**GitHub Repository:** https://github.com/moldirtolegenova-cb/research_methods-

**Research Area:** Cybersecurity, Network Traffic Analysis, Machine Learning

**Research Type:** Quantitative experimental research.

## 2. Research Problem

Distributed Denial of Service (DDoS) attacks can disrupt network services by overwhelming their resources with malicious traffic. Machine learning classifiers can be used to classify network traffic flows, but their measured performance may vary depending on the classifier and the features used.

A controlled comparison is needed to examine these factors under a consistent evaluation protocol.

The current repository implements a reproducible pipeline using synthetic data. A representative labeled network traffic dataset is required for meaningful empirical conclusions.

## 3. Research Object

Flow-based network traffic records containing benign and DDoS traffic.

## 4. Research Subject

The effect of machine learning classifier choice and feature-set selection on the performance of flow-based DDoS attack detection.

## 5. Research Objective

To implement a reproducible experimental pipeline and compare four machine learning classifiers across two predefined feature sets using consistent evaluation conditions.

## 6. Research Question

How do different machine learning classifiers and feature selection strategies affect the performance of flow-based DDoS attack detection?

## 7. Research Hypotheses

**Null Hypothesis (H0):** Under the same dataset, preprocessing procedure, train-test split, and evaluation protocol, the evaluated classifiers and feature sets do not produce a practically meaningful difference in F1-score.

**Alternative Hypothesis (H1):** At least one evaluated classifier or feature set produces a practically meaningful difference in F1-score.

A minimum F1-score difference of 0.02 is defined as the practical significance threshold. This threshold is a planned comparison criterion, not a claim of statistical significance.

Statistical significance cannot be established using the current ten-record synthetic sample. A full study should use representative data and an appropriate repeated or cross-validation protocol with a suitable statistical comparison.

## 8. Independent Variables

### Classifier Type

Four classifiers are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### Feature Set

Two predefined feature sets are compared:

- **Full:** `flow_duration`, `total_fwd_packets`, `total_bwd_packets`, `flow_bytes_per_sec`, and `packet_rate`.
- **Selected:** `flow_duration`, `total_fwd_packets`, and `flow_bytes_per_sec`.

The selected subset is predefined in the configuration. It is not claimed to be an optimally selected subset.

## 9. Dependent Variables

**Primary dependent variable:**

- F1-score.

**Secondary dependent variables:**

- Precision
- Recall
- Accuracy
- False-positive rate (FPR)

F1-score is the primary metric because it combines precision and recall. Recall measures the proportion of positive cases correctly identified. FPR measures the proportion of negative cases incorrectly classified as positive.

## 10. Controlled Variables

The following conditions are kept consistent across model configurations:

- Dataset and target-label definitions.
- Input feature definitions.
- Train-test partition.
- Random seed.
- Evaluation metrics.
- Baseline definition.
- Guardrail thresholds.
- Python and dependency versions, as documented for each run.

Model-specific algorithm settings may differ because the classifiers have different parameters.

## 11. Dataset and Sampling

The repository includes `data/sample/flows.csv`, which contains ten synthetic records. The sample is used only to validate the pipeline.

The target column is `label`. In the included sample, `0` represents benign traffic and `1` represents DDoS traffic.

A stratified train-test split is created using a configured test size of 0.40 and random seed of 42.

The small synthetic dataset is not suitable for estimating real-world detection performance. A full experiment requires a representative, documented, labeled network traffic dataset.

For the full study, the dataset source, license, label mapping, preprocessing, missing-value handling, and any exclusions should be documented.

## 12. Experimental Procedure

1. Load the YAML experiment configuration.
2. Load the dataset from the configured path.
3. Validate the target column and check that both classes are present.
4. Convert the configured positive label to the positive binary class.
5. Validate the required features and check for missing or infinite feature values.
6. Create one stratified train-test split using the configured random seed.
7. Train each of the four classifiers on the full feature set.
8. Train each classifier on the selected feature set.
9. Calculate accuracy, precision, recall, F1-score, and false-positive rate.
10. Identify the baseline configuration: Logistic Regression with the full feature set.
11. Calculate the F1-score difference between every configuration and the baseline.
12. Check whether each configuration meets the recall and FPR guardrails.
13. Save the results to `results/benchmark_results.csv`.

## 13. Evaluation Criteria

**Primary metric:** F1-score.

**Practical difference threshold:** 0.02 F1-score.

**Baseline:** Logistic Regression with the full feature set.

**Guardrails:**

- Recall must be at least 0.90.
- False-positive rate must be no more than 0.05.

Guardrail results indicate whether a configuration meets the predefined targets on the current test split. They do not establish general performance.

## 14. Reproducibility

The project uses:

- A YAML configuration file.
- Pinned Python package versions.
- A fixed random seed.
- A consistent train-test split across model configurations.
- A benchmark script.
- A machine-readable CSV output.

To reproduce the pipeline, install the dependencies and run:

```bash
python src/benchmark.py --config configs/config.yaml
```

## 15. Ethical and Practical Considerations

The included data are synthetic and do not contain real user network records. When using a real dataset, its license, privacy implications, and permitted use must be reviewed.

The results should not be used alone to make operational security decisions.

## 16. Limitations

- The sample contains only ten synthetic records.
- Synthetic patterns may be much easier to classify than real network traffic.
- One train-test split can produce unstable performance estimates.
- The current experiment does not establish statistical significance.
- The selected feature subset is predefined rather than selected through a separate feature-selection experiment.
- Results cannot be generalized to real-world DDoS detection.

A future study should use representative data, repeated evaluation or cross-validation, documented preprocessing, and appropriate statistical analysis.

## 17. Expected Research Contribution

The current contribution is a reproducible benchmark pipeline that compares four classifiers across two predefined feature sets and reports consistent classification metrics.

Empirical conclusions about which model performs best will require evaluation on a suitable real-world or appropriately documented research dataset.
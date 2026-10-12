# Methodology Passport Card

## 1. Project Information

**Title:** Comparative Evaluation of Machine Learning Models for Flow-Based DDoS Attack Detection

**Authors:** [Kudaibergen Nazerke], [Tolegenova Moldir]

**GitHub Repository:** [https://github.com/moldirtolegenova-cb/research_methods-/tree/main]

**Research Area:** Cybersecurity, Network Traffic Analysis, Machine Learning

## 2. Research Question

How do different machine learning classifiers and feature selection strategies affect the performance of flow-based DDoS attack detection?

## 3. Research Type

This study uses a **quantitative experimental research design**. Numerical classification metrics are used to compare machine learning models under controlled experimental conditions.

## 4. Research Hypotheses

**Null Hypothesis (H0):** Under the same dataset, preprocessing procedure, train-test split, and evaluation protocol, the evaluated machine learning classifiers and feature sets do not produce a practically meaningful difference in F1-score.

**Alternative Hypothesis (H1):** Under the same dataset, preprocessing procedure, train-test split, and evaluation protocol, at least one evaluated classifier or feature set produces a practically meaningful difference in F1-score.

For the final evaluation, practical significance will be assessed using a pre-specified minimum F1-score difference of 0.02. Statistical significance will be assessed using an appropriate paired statistical procedure when repeated or cross-validation results are available.

## 5. Variables Matrix

### Independent Variables

* Classifier type: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
* Feature set: full feature set versus a predefined selected feature subset.

### Dependent Variables

* Primary metric: F1-score.
* Secondary metrics: precision, recall, accuracy, and false-positive rate (FPR).

### Controlled Variables

* Dataset and label definitions.
* Data cleaning and preprocessing procedure.
* Fixed training and test partition or identical cross-validation folds.
* Random seed and evaluation protocol.
* Python and library versions.
* Hardware and operating system, documented for each experiment.
* Classifier settings, except for the classifier-specific differences required by the experiment.

## 6. Primary Metric, Guardrails, and Baseline

**Primary Metric:** F1-score, which balances precision and recall and is suitable for comparing detection performance.

**Guardrail Metrics:**

* Recall should remain at or above 0.90.
* False-positive rate should remain at or below 0.05.
* No critical pipeline errors or missing evaluation results.

These thresholds are pre-specified research targets, not guaranteed outcomes.

**Baseline:** Logistic Regression using the full feature set. All other classifiers and the selected-feature configuration will be compared against this baseline.

## 7. Experimental Procedure

1. Load and validate the flow-based network traffic dataset.
2. Apply the same preprocessing and label mapping to all experiments.
3. Use the same train-test partition or cross-validation folds.
4. Train each classifier using the full and selected feature sets.
5. Calculate F1-score, precision, recall, accuracy, and FPR.
6. Save the configuration and results in machine-readable CSV format.
7. Compare the results with the baseline and evaluate the hypotheses.

## 8. Reproducibility and Limitations

The repository will include pinned dependencies, an experiment configuration file, a runnable benchmark script, and a small synthetic sample dataset for pipeline testing. The synthetic sample is intended only to verify execution and must not be treated as evidence of real-world DDoS detection performance.

The final evaluation requires a suitable labeled flow-based network traffic dataset and an evaluation protocol that avoids data leakage.

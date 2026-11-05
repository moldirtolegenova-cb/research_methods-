
# Domain Model: Flow-Based DDoS Attack Detection

## 1. Research Domain

This research evaluates machine learning classifiers
for detecting Distributed Denial of Service (DDoS)
attacks using network flow-based features.

The system receives network flow records and predicts
whether the observed traffic is benign or malicious.

The task is formulated as binary classification.

## 2. Dataset Definition

Let the dataset be:

\[
D = \{(x_i,y_i)\}_{i=1}^{N}
\]

where:

- \(N\) is the number of observations.
- \(x_i \in \mathbb{R}^{d}\) is an input feature vector.
- \(d\) is the number of input features.
- \(y_i \in \{0,1\}\) is the actual class label.

Class definitions:

- \(y_i=0\): Benign traffic.
- \(y_i=1\): DDoS attack traffic.

The input features include:

- Flow duration.
- Total forward packets.
- Total backward packets.
- Flow bytes per second.
- Packet rate.

The current sample dataset contains ten synthetic
observations. Therefore, the classification results
are preliminary and cannot establish real-world
DDoS detection performance.

## 3. Binary Classification Model

A classifier estimates the probability of an attack:

\[
\hat{p}_i = f_{\theta}(x_i)
\]

where:

- \(f_{\theta}\) is the classification model.
- \(\theta\) represents the model parameters.
- \(\hat{p}_i\) is the predicted probability of an attack.

The predicted class is:

\[
\hat{y}_i =
\begin{cases}
1, & \hat{p}_i \geq 0.5 \\
0, & \hat{p}_i < 0.5
\end{cases}
\]

The experiment compares:

1. Logistic Regression.
2. Decision Tree.
3. Random Forest.
4. XGBoost.

## 4. Binary Cross-Entropy Loss Function

The binary classification objective is defined as:

\[
L(\theta) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right]
\]

To avoid undefined logarithms, predicted probabilities
are interpreted within the interval:

\[
\hat{p}_i \in [\epsilon,1-\epsilon]
\]

where \(\epsilon\) is a small positive constant.

The loss function measures the difference between
actual labels and predicted probabilities.

This is a formal binary classification objective.
It is not claimed that every evaluated classifier
directly optimizes this exact loss function.

## 5. Classification Evaluation Metrics

### 5.1 Accuracy

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### 5.2 Precision

\[
Precision = \frac{TP}{TP+FP}
\]

### 5.3 Recall

\[
Recall = \frac{TP}{TP+FN}
\]

### 5.4 F1-score

\[
F1 = 2 \cdot
\frac{Precision \cdot Recall}
{Precision+Recall}
\]

### 5.5 False Positive Rate

\[
FPR = \frac{FP}{FP+TN}
\]

where:

- TP: True Positive.
- TN: True Negative.
- FP: False Positive.
- FN: False Negative.

If a denominator is zero, the corresponding
metric is reported according to the implementation's
zero-division policy.

## 6. Computational Performance Metrics

### 6.1 Inference Latency

\[
Latency = t_{end}-t_{start}
\]

Latency is measured in milliseconds:

\[
Latency_{ms} =
(t_{end}-t_{start})\times 1000
\]

### 6.2 Throughput

\[
Throughput = \frac{B}{T}
\]

where:

- \(B\) is the number of processed samples.
- \(T\) is the elapsed time in seconds.

The unit is samples per second.

### 6.3 Error Rate

\[
ErrorRate = \frac{E}{R}
\]

where:

- \(E\) is the number of failed evaluation runs.
- \(R\) is the total number of evaluation runs.

The error rate is used as a performance guardrail.

## 7. Statistical Analysis

Warm-up measurements are excluded before
calculating final performance statistics.

### 7.1 Median

\[
Median(X)=Q_{0.50}(X)
\]

### 7.2 Percentiles

\[
p95=Q_{0.95}(X)
\]

\[
p99=Q_{0.99}(X)
\]

These percentiles describe the upper tail
of the latency distribution.

### 7.3 Bootstrap Confidence Interval

Bootstrap resampling is performed with replacement.

For a statistic \(\hat{\theta}^{*}\), the 95%
percentile confidence interval is:

\[
CI_{95\%} =
[
Q_{0.025}(\hat{\theta}^{*}),
Q_{0.975}(\hat{\theta}^{*})
]
\]

The experiment uses:

- Bootstrap iterations: 5000.
- Random seed: 42.
- Confidence level: 95%.

The bootstrap interval describes uncertainty in
the estimated statistic under the observed sample.

Repeated cross-validation folds may share observations
and should not be interpreted as fully independent
experimental samples.

## 8. Performance Guardrails

The predefined classification targets are:

\[
Recall \geq 0.90
\]

\[
FPR \leq 0.05
\]

These values are target thresholds, not guaranteed
experimental outcomes.

The computational guardrail is:

\[
ErrorRate = \frac{E}{R}
\]

A lower error rate indicates more reliable
benchmark execution.

## 9. Experimental Comparison

The baseline model is Logistic Regression.

The alternative models are:

- Decision Tree.
- Random Forest.
- XGBoost.

The comparison considers:

- F1-score.
- False Positive Rate.
- Median inference latency.
- p95 latency.
- p99 latency.
- Throughput.
- Error Rate.

The preferred configuration should balance
classification quality and computational efficiency.

## 10. Mathematical Notation Table

| Symbol | Definition |
|---|---|
| \(D\) | Dataset |
| \(N\) | Number of observations |
| \(d\) | Number of features |
| \(x_i\) | Input feature vector |
| \(y_i\) | Actual binary label |
| \(\hat{y}_i\) | Predicted binary label |
| \(\hat{p}_i\) | Predicted attack probability |
| \(\theta\) | Model parameters |
| \(L(\theta)\) | Binary Cross-Entropy Loss |
| TP | True Positive |
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |
| \(t_{start}\) | Inference start time |
| \(t_{end}\) | Inference end time |
| \(B\) | Number of processed samples |
| \(T\) | Elapsed time |
| \(E\) | Failed evaluation runs |
| \(R\) | Total evaluation runs |
| \(Q_p\) | Percentile at probability \(p\) |
| \(CI_{95\%}\) | 95% Confidence Interval |

## 11. Limitations

The current classification dataset contains only
ten synthetic observations.

Although repeated stratified cross-validation
is used, the small dataset cannot provide
a reliable estimate of performance on real
network traffic.

The benchmark measurements also depend on
hardware, software versions and execution conditions.

Further research should use a larger, representative,
labeled network traffic dataset.

Results should therefore be interpreted as
preliminary experimental findings.

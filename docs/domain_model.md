
# Domain Model: Flow-Based DDoS Detection

## 1. Research Domain

The research focuses on detecting Distributed Denial of Service
(DDoS) attacks using machine learning classifiers trained on
network flow features.

The system receives network flow records and classifies each
record as either legitimate traffic or malicious traffic.

## 2. Input Data

Let the dataset be:

\[
D = \{(x_i,y_i)\}_{i=1}^{N}
\]

where:

- \(x_i \in \mathbb{R}^{d}\) is a feature vector.
- \(y_i \in \{0,1\}\) is the binary class label.
- \(N\) is the number of observations.
- \(d\) is the number of selected features.

Class definitions:

- \(y_i=0\): legitimate network traffic.
- \(y_i=1\): DDoS attack traffic.

## 3. Binary Classification Model

Each classifier estimates:

\[
\hat{p}_i = f_{\theta}(x_i)
\]

where:

- \(f_{\theta}\) is the selected machine learning model.
- \(\theta\) represents model parameters.
- \(\hat{p}_i\) is the predicted probability of an attack.

The final prediction is:

\[
\hat{y}_i =
\begin{cases}
1, & \hat{p}_i \geq 0.5 \\
0, & \hat{p}_i < 0.5
\end{cases}
\]

## 4. Binary Cross-Entropy Loss

The objective function for binary classification is:

\[
L(\theta) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right]
\]

The loss measures the difference between actual labels
and predicted attack probabilities.

For models that do not directly optimize binary cross-entropy,
the same loss can be used as a common evaluation objective
when probability estimates are available.

## 5. Evaluation Metrics

### Accuracy

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

\[
Precision = \frac{TP}{TP+FP}
\]

### Recall

\[
Recall = \frac{TP}{TP+FN}
\]

### F1-score

\[
F1 = 2 \cdot
\frac{Precision \cdot Recall}
{Precision+Recall}
\]

### False Positive Rate

\[
FPR = \frac{FP}{FP+TN}
\]

### Inference Latency

\[
Latency = t_{end}-t_{start}
\]

Latency is measured in milliseconds.

### Throughput

\[
Throughput = \frac{B}{T}
\]

where \(B\) is the number of processed samples
and \(T\) is the elapsed time in seconds.

Throughput is measured in samples per second.

### Error Rate

\[
ErrorRate = \frac{E}{R}
\]

where \(E\) is the number of failed evaluation runs
and \(R\) is the total number of evaluation runs.

## 6. Statistical Analysis

The median is:

\[
Median(X) = Q_{0.50}(X)
\]

The 95th and 99th percentiles are:

\[
p95 = Q_{0.95}(X)
\]

\[
p99 = Q_{0.99}(X)
\]

The bootstrap confidence interval is estimated using
resampling with replacement.

For a bootstrap estimate \(\hat{\mu}^{*}\):

\[
CI_{95\%} =
[Q_{0.025}(\hat{\mu}^{*}),
Q_{0.975}(\hat{\mu}^{*})]
\]

The random seed is fixed at 42.

## 7. Mathematical Notation Table

| Symbol | Definition |
|---|---|
| \(D\) | Dataset |
| \(N\) | Number of observations |
| \(d\) | Number of input features |
| \(x_i\) | Input feature vector |
| \(y_i\) | Actual binary label |
| \(\hat{y}_i\) | Predicted binary label |
| \(\hat{p}_i\) | Predicted attack probability |
| \(\theta\) | Model parameters |
| \(L(\theta)\) | Loss function |
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

## 8. Experimental Comparison

The research compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The models are evaluated using classification quality,
computational performance and error-rate guardrails.

The preferred model should balance a high F1-score,
a low false positive rate, low inference latency
and acceptable computational efficiency.

## 9. Limitations

The reliability of the conclusions depends on dataset size,
class balance, data quality and evaluation methodology.

Results from a small sample dataset are preliminary
and should not be generalized to production network traffic.

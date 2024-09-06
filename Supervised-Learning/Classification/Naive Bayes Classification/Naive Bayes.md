# Naive Bayes Classification

Naive Bayes classification algorithm is a supervised learning algorithm based on Bayes' Theorem, used for classification tasks in machine learning. It assumes that the features are conditionally independent, which simplifies the calculation of probabilities and makes it efficient for high-dimensional data.


\[
P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
\]

Where:
- \( P(C \mid X) \): Posterior probability of class \( C \) given the features \( X \).
- \( P(X \mid C) \): Likelihood of features \( X \) given class \( C \).
- \( P(C) \): Prior probability of class \( C \).
- \( P(X) \): Evidence, which is constant for all classes.

The model classifies data by selecting the class \( C \) that maximizes \( P(C \mid X) \).

### Assumption: Conditional Independence
Naive Bayes assumes that all features \( X_i \) are conditionally independent given the class \( C \). This simplifies the likelihood term:

\[
P(X \mid C) = P(X_1 \mid C) \cdot P(X_2 \mid C) \cdot \dots \cdot P(X_n \mid C)
\]

## Types of Naive Bayes
There are several variations of the Naive Bayes classifier depending on the type of data:
- **Gaussian Naive Bayes**: Used when features are continuous and normally distributed.
- **Multinomial Naive Bayes**: Suitable for discrete data, such as word counts in a document.
- **Bernoulli Naive Bayes**: Works with binary data, where features are binary (0 or 1).

### Gaussian Naive Bayes
In the case of continuous features (e.g., numerical data), we assume the features follow a Gaussian (normal) distribution:

\[
P(X_i \mid C) = \frac{1}{\sqrt{2 \pi \sigma_C^2}} \exp \left( -\frac{(X_i - \mu_C)^2}{2 \sigma_C^2} \right)
\]

Where \( \mu_C \) and \( \sigma_C^2 \) are the mean and variance of the feature \( X_i \) for class \( C \).

## Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/username/naive-bayes-classification.git
cd naive-bayes-classification
pip install -r requirements.txt



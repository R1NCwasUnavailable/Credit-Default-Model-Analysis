# Comparative Analysis: Credit Card Default Prediction

## 1. Introduction
The objective of this study is to predict the likelihood of credit card default using the UCI Credit Card Clients dataset. We compare multiple machine learning algorithms ranging from baseline linear models to complex ensemble methods. The primary focus is on handling class imbalance and minimizing financial risk by optimizing for Recall and F1-score.

## 2. Methodology

### 2.1 Dataset & Preprocessing
The dataset contains 30,000 observations. The target variable is `default payment next month`.
-   **Class Imbalance**: The dataset is imbalanced (approx. 22% default). We address this by using **stratified splitting** and applying `class_weight='balanced'` in our cost-sensitive models (Logistic Regression, SVM, Decision Tree, Random Forest).
-   **Feature Scaling**: continuous features (`LIMIT_BAL`, `AGE`, `BILL_AMT`, `PAY_AMT`) are scaled using `StandardScaler` to ensure distance-based models (SVM, k-NN) perform optimally. Tree-based models are robust to scaling but receive the same processed data for consistency.
-   **Encoding**: Categorical variables (`EDUCATION`, `MARRIAGE`) are grouped to reduce noise (merging 'unknown' categories).

### 2.2 Models
We evaluate three categories of models:
1.  **Baseline**:
    -   *Logistic Regression*: Provides a linear baseline and interpretability (coefficients).
    -   *Gaussian Naive Bayes*: A simple probabilistic baseline.
2.  **Non-linear**:
    -   *k-Nearest Neighbors (k-NN)*: Captures local structure but is computationally expensive.
    -   *Decision Tree*: Captures non-linear splits but is prone to overfitting.
    -   *Support Vector Machine (SVM)*: Effective in high-dimensional spaces (using RBF kernel).
3.  **Ensemble**:
    -   *Random Forest*: Reduces variance of decision trees via bagging.
    -   *Gradient Boosting*: Reduces bias by sequentially correcting errors.
    -   *Voting Classifier*: Combines predictions from LR, DT, and SVM.

## 3. Evaluation Metrics
Given the domain (Credit Risk), we prioritize:
-   **Recall**: Capturing as many true defaults as possible (minimizing False Negatives). Missing a defaulter is costly.
-   **F1-Score**: Harmonic mean of Precision and Recall, providing a balance.
-   **ROC-AUC**: Measures the model's ability to distinguish between classes across thresholds.

## 4. Results
(This section will be populated after model training)

## 5. Discussion

### 5.1 Bias-Variance Tradeoff
-   **Logistic Regression** & **Naive Bayes** generally have high bias but low variance. They may underfit complex relationships.
-   **Decision Trees** have low bias but high variance (overfitting).
-   **Ensembles (RF, GBM)** aim to optimize this tradeoff. RF reduces variance, GBM reduces bias.

### 5.2 Scaling Effects
-   Scaling is critical for **k-NN** (distance calculation) and **SVM** (margin maximization) and **Logistic Regression** (regularization).
-   Tree-based models are invariant to monotonic transformations, so scaling is not strictly necessary but harmless.

### 5.3 False Positives vs. False Negatives
-   **False Negative (Type II Error)**: Predicting "No Default" when the customer actually defaults. **High Cost** (Financial Loss).
-   **False Positive (Type I Error)**: Predicting "Default" when the customer pays. **Opportunity Cost** (Lost interest/customer friction).
-   Our strategy (`class_weight='balanced'`) penalizes False Negatives more heavily, likely increasing Recall at the expense of Precision.

## 6. Conclusion
(To be added)

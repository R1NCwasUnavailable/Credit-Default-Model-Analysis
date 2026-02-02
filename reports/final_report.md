# Comparative Analysis: Credit Card Default Prediction

## 1. Introduction
The objective of this study was to predict the likelihood of credit card default using the UCI Credit Card Clients dataset. We compared multiple machine learning algorithms ranging from baseline linear models to complex ensemble methods, addressing class imbalance and prioritizing Recall and ROC-AUC.

## 2. Methodology
-   **Dataset**: 30,000 observations. 22% default rate.
-   **Preprocessing**: Stratified Train-Test Split (80/20). Standard Scaling for continuous variables. Categorical grouping for Education/Marriage.
-   **Models**: 
    -   *Baselines*: Logistic Regression, Gaussian NB.
    -   *Non-linear*: k-NN, Decision Tree, SVM (RBF).
    -   *Ensembles*: Random Forest, Gradient Boosting, Voting Classifier.
-   **Optimization**: Due to computational constraints, SVM and Voting Classifier were trained on a stratified subset (5,000 samples) to allow for probability calibration (5-fold internal CV) within reasonable time. All other models used the full training set (24,000 samples).

## 3. Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.819** | 0.67 | 0.36 | 0.47 | **0.779** |
| Random Forest | 0.813 | 0.65 | 0.34 | 0.45 | 0.758 |
| Gaussian NB | 0.752 | 0.38 | 0.72 | 0.50 | 0.725 |
| SVM (Subset) | 0.780 | 0.46 | 0.59 | **0.52** | 0.742 |
| Voting Classifier | 0.766 | 0.43 | 0.41 | 0.42 | 0.712 |
| Logistic Regression | 0.680 | 0.30 | 0.74 | 0.46 | 0.708 |
| k-NN | 0.793 | 0.58 | 0.34 | 0.43 | 0.702 |
| Decision Tree | 0.729 | 0.38 | 0.39 | 0.39 | 0.606 |

### Visualizations
![ROC Curves](reports/roc_curves.png)

## 4. Discussion

### 4.1 Best Performing Model
**Gradient Boosting** achieved the highest discrimination ability (ROC-AUC 0.779) and Accuracy (81.9%). However, its Recall (0.36) is lower than baseline models.
**SVM** (trained on a subset) demonstrated a strong balance, achieving the highest **F1-Score (0.52)** and good ROC-AUC (0.742). It captured 59% of defaulters (Recall) while maintaining decent precision.

### 4.2 Bias-Variance Tradeoff
-   **Decision Tree**: Low ROC-AUC (0.606) indicates high variance/overfitting. It failed to generalize compared to ensembles.
-   **Logistic Regression / Naive Bayes**: High Recall (0.74 / 0.72) but low Precision. They "over-predict" default due to class weights/probabilistic assumptions (high bias), which is "safer" for risk detecting but yields many False Positives.
-   **Ensembles**: Successfully reduced variance (RF) and bias (GB), leading to better ranking (AUC).

### 4.3 Scaling Effects
Scaling was crucial for k-NN and SVM. Without it, dominance of `LIMIT_BAL` (large values) would distort distance calculations.

### 4.4 False Positives vs. False Negatives
-   **Risk Aversion**: If the bank wants to catch *all* potential defaults, **Gaussian NB** or **Logistic Regression** are best (Recall > 70%), despite high False Alarm rate.
-   **Balanced Approach**: **SVM** offers the best trade-off (F1 0.52).
-   **Precision Focus**: **Gradient Boosting** is best if the cost of investigating a False Alarm is high.

## 5. Conclusion
Gradient Boosting is the most robust model for ranking customers by risk (AUC). However, for a practical credit default system where missing a defaulter is costly, the SVM (or a tuned Logistic Regression) might be preferred due to higher Recall/F1.

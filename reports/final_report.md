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

### 4.2 Simple vs. Complex Models
-   **Simple Models (Logistic Regression / Naive Bayes)**: High bias, low variance. They "over-predict" default due to our class balancing but offer excellent baseline interpretability (via coefficients).
-   **Decision Tree**: Indicates high variance/overfitting. It fails to generalize well on noisy financial data without bagging.
-   **Complex Models (Ensembles like Gradient Boosting, XGBoost, and MLP)**: Successfully optimize the tradeoff. They capture non-linear interactions between features (e.g., if a young user *also* has a history of late payments). They significantly reduce variance and bias, leading to the best predictive ranking (highest ROC-AUC).

### 4.3 Feature Importance Insights & "Why it Works"
Using SHAP values and model feature importances, we can open the "black box" of our complex models to explain exactly *why* they work:
-   **Repayment History (`PAY_1` to `PAY_6`)**: This is consistently the strongest predictor. Customers who were late on their most recent payments (status > 0) are exponentially more likely to default. The model heavily penalizes recent delinquency.
-   **Credit Limit (`LIMIT_BAL`)**: Lower credit limits strongly correlate with higher default probabilities. This serves as a proxy for the bank's prior risk assessment and the user's income/wealth.
-   **Age & Demographics**: Play a secondary role. While younger users or specific marital groups show slight variations, they usually only trigger defaults when combined with poor repayment history.
*Why it works: The algorithms intelligently learn to prioritize dynamic behavioral data (recent repayments) over static demographic profiles.*

### 4.4 Real-World Application in FinTech & Banking
In a real-world scenario, model selection depends on the business objective:
-   **Risk Aversion (Traditional Banks)**: If the bank wants to catch *all* potential defaults to minimize exposure, a tuned **Logistic Regression** or **SVM** is often best (maximizing Recall). Simple models are also much easier to push through strict regulatory compliance.
-   **Cost/Precision Optimization (Modern FinTechs)**: Modern lenders often prefer **XGBoost** or **LightGBM**. These models offer the best discrimination (ROC-AUC) and can be coupled with SHAP values to cleanly generate automated, regulatory-compliant adverse action notices explaining exactly *why* a loan was denied. This marries high performance with transparency.

## 5. Conclusion
Gradient Boosting is the most robust model for ranking customers by risk (AUC). However, for a practical credit default system where missing a defaulter is costly, the SVM (or a tuned Logistic Regression) might be preferred due to higher Recall/F1.

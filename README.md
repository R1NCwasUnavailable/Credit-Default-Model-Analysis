# Credit Default Model Analysis

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning analysis to predict credit card default using the **UCI Credit Card Clients dataset**. This project compares baseline linear models, non-linear algorithms, and advanced ensemble methods (like XGBoost, LightGBM, CatBoost), with a focus on **Recall** and **ROC-AUC** to address financial risk and class imbalance. It implements automated hyperparameter tuning via `RandomizedSearchCV` for state-of-the-art tree models.

## 📊 Key Results

After incorporating and tuning advanced ensemble models, Gradient Boosted Trees consistently dominated the classic baseline models. Here are the top performers by ROC-AUC on the test set:

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | 81.71% | 0.462 | **0.7785** |
| **XGBoost** | 81.85% | 0.464 | **0.7775** |
| **CatBoost** | 76.36% | **0.538** | **0.7767** |
| **LightGBM** | 75.75% | 0.530 | 0.7767 |
| **Random Forest** | 78.73% | 0.543 | 0.7757 |

> **Insight**: While Gradient Boosting and XGBoost provide the best overall ranking capability (AUC), models like CatBoost, LightGBM, and Random Forest attained significantly higher F1-Scores. This indicates they successfully identified a much larger proportion of actual defaults (higher Recall) at the expense of false positives, making them exceptionally strong candidates for practical risk mitigation.

## 🚀 Project Structure

```
├── data/               # Dataset storage
│   ├── raw/            # Original UCI dataset
│   └── processed/      # Cleaned & split CSVs (X_train, y_train, etc.)
├── notebooks/          # Exploratory analysis
│   └── 01_EDA_and_Modeling.ipynb # Interactive EDA and modeling demo
├── reports/            # Generated figures, CSVs, and final analysis
│   ├── figures/        # Confusion matrices for all 18 models
│   ├── model_comparison.csv # Raw evaluation metrics
│   ├── roc_curves.png  # Consolidated ROC curves
│   └── final_report.md # Detailed research analysis
├── src/                # Modular source code
│   ├── data_loader.py  # Data ingestion
│   ├── models.py       # Model definitions and hyperparameter grids
│   ├── preprocessing.py# Data cleaning & scaling
│   └── evaluation.py   # Metric calculation
├── main.py             # Master script to run the full training pipeline
└── requirements.txt    # Project dependencies (includes xgboost, lightgbm, catboost)
```

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/R1NCwasUnavailable/Credit-Default-Model-Analysis.git
    cd Credit-Default-Model-Analysis
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis**
    This script will download the data (if missing), preprocess it, automatically tune specific models using `RandomizedSearchCV`, train all 18 models, and generate the reporting artifacts.
    ```bash
    python main.py
    ```

4.  **View Results**
    Check the `reports/` directory for `model_comparison.csv` and generated plots.

## 📈 Visualizations

**ROC Curves Comparison**
![ROC Curves](reports/roc_curves.png)

## 📝 Methodology
-   **Preprocessing**: Stratified 80/20 split, Standard Scaling (for distance-based models like SVM/k-NN), Class Weight Balancing.
-   **Baseline & Linear Models**: Logistic Regression, Naive Bayes, Ridge Classifier, SGD Classifier, LDA.
-   **Non-Linear Models**: k-NN, Decision Tree, SVM.
-   **Ensemble Models**: Random Forest, Gradient Boosting, AdaBoost, Extra Trees, XGBoost, LightGBM, CatBoost, Voting Classifier, Stacking Classifier.
-   **Optimization**: 
    - `RandomizedSearchCV` was used extensively to optimize hyperparameters for tree-based ensemble models with 10 iterations and 3-fold cross-validation.
    - Stratified subsampling was used for computationally expensive models (like SVM) to enable probability calibration within reasonable training time.

## 🤝 Contributing
Feel free to open issues or submit pull requests with additional feature engineering concepts or broader tuning grids.

## 📄 License
MIT

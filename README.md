# Credit Default Model Analysis

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning analysis to predict credit card default using the **UCI Credit Card Clients dataset**. This project compares baseline linear models, non-linear algorithms, and ensemble methods, with a focus on **Recall** and **ROC-AUC** to address financial risk and class imbalance.

## 📊 Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **81.9%** | 0.47 | **0.78** |
| Random Forest | 81.3% | 0.45 | 0.76 |
| **SVM (Optimized)** | 78.0% | **0.52** | 0.74 |
| Gaussian NB | 75.2% | 0.50 | 0.72 |
| Logistic Regression | 68.0% | 0.46 | 0.71 |

> **Insight**: Gradient Boosting provides the best overall ranking capability (AUC), while SVM (trained on a balanced subset) offers the best balance of Precision and Recall (F1), making it a strong candidate for practical default detection.

## 🚀 Project Structure

```
├── data/               # Dataset storage
│   ├── raw/            # Original UCI dataset
│   └── processed/      # Cleaned & split CSVs (X_train, y_train, etc.)
├── notebooks/          # Exploratory analysis
│   └── 01_EDA_and_Modeling.ipynb # Interactive EDA and modeling demo
├── reports/            # Generated figures, CSVs, and final analysis
│   ├── figures/        # Confusion matrices and plots
│   └── final_report.md # Detailed research analysis
├── src/                # Modular source code
│   ├── data_loader.py  # Data ingestion
│   ├── models.py       # Model definitions
│   ├── preprocessing.py# Data cleaning & scaling
│   └── evaluation.py   # Metric calculation
├── main.py             # Master script to run the full pipeline
└── requirements.txt    # Project dependencies
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
    This script will download the data (if missing), preprocess it, train all models, and generate the report.
    ```bash
    python main.py
    ```

4.  **View Results**
    Check the `reports/` directory for the `model_comparison.csv` and generated plots.

5.  **Interactive Exploration**
    Open the Jupyter Notebook to explore data distributions and run quick experiments:
    ```bash
    jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
    ```

## 📈 Visualizations

**ROC Curves Comparison**
![ROC Curves](reports/roc_curves.png)

## 📝 Methodology
-   **Preprocessing**: Stratified 80/20 split, Standard Scaling (for SVM/k-NN), Class Weight Balancing.
-   **Models**: Logistic Regression, Naive Bayes, k-NN, Decision Tree, SVM, Random Forest, Gradient Boosting, Voting Classifier.
-   **Optimization**: Stratified subsampling used for SVM to enable probability calibration within reasonable training time.

## 🤝 Contributing
Feel free to open issues or submit pull requests if you have suggestions for improving feature engineering or model hyperparameters.

## 📄 License
MIT

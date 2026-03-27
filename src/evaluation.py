from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    ks_stat = "N/A"
    
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        # Calculate KS Statistic
        prob_pos = y_prob[y_test == 1]
        prob_neg = y_prob[y_test == 0]
        if len(prob_pos) > 0 and len(prob_neg) > 0:
            ks_stat, _ = ks_2samp(prob_neg, prob_pos)
    else:
        # Some models might not support probability (e.g. SVM without probability=True)
        # But we set probability=True for SVM in models.py
        try:
            y_prob = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
             roc_auc = "N/A"

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc,
        'KS-Stat': ks_stat
    }
    return metrics

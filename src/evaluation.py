from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
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
        'ROC-AUC': roc_auc
    }
    return metrics

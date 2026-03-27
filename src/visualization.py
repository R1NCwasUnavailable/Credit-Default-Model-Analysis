import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_roc_curves(models_dict, X_test, y_test, output_path=None):
    """
    Plots ROC curves for multiple models.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(output_path)
    # plt.show() # Commented out for non-interactive run if needed

def plot_confusion_matrices(models_dict, X_test, y_test, output_dir='reports/figures'):
    """
    Plots confusion matrices for each model.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cm_{name.replace(' ', '_')}.png")
        plt.close()

def plot_shap_values(model, X_test, model_name, output_dir='reports/figures'):
    """
    Generates SHAP summary plots for a given model.
    """
    import os
    import shap
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for non-tree models
            kmeans_data = shap.kmeans(X_test, 10)
            explainer = shap.KernelExplainer(model.predict_proba, kmeans_data)
            X_test = shap.sample(X_test, 100) # sample to speed up
            
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_{model_name.replace(' ', '_')}.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate SHAP for {model_name}: {e}")

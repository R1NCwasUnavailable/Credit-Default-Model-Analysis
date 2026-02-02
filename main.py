import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import get_baseline_models, get_nonlinear_models, get_ensemble_models
from src.evaluation import evaluate_model
from src.visualization import plot_roc_curves, plot_confusion_matrices

def main():
    # 1. Load Data
    print("Loading Data...")
    df = load_data()
    if df is None:
        return
    
    # 2. Preprocess
    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 3. Define Models
    baseline = get_baseline_models()
    nonlinear = get_nonlinear_models()
    ensemble = get_ensemble_models(baseline, nonlinear)
    
    all_models = {**baseline, **nonlinear, **ensemble}
    
    # 4. Train & Evaluate
    results = []
    print("\nTraining & Evaluating Models...")
    
    # Create reports directory
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    for name, model in all_models.items():
        print(f" - Training {name}...")
        
        # Optimization for time-consuming SVM models
        # Voting Classifier also includes SVM so we subsample for it too if it's slow
        if name in ['SVM', 'Voting Classifier'] and len(X_train) > 5000:
             print(f"   [Optimization] Subsampling training data for {name} to 5000 samples for speed...")
             # Stratified subsample
             X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=5000, stratify=y_train, random_state=42)
             model.fit(X_sub, y_sub)
        else:
             model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"   Done. ROC-AUC: {metrics['ROC-AUC']}")

    # 5. Compile Results
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df)
    results_df.to_csv('reports/model_comparison.csv', index=False)
    
    # 6. Visualization
    print("\nGenerating Plots...")
    plot_roc_curves(all_models, X_test, y_test, output_path='reports/roc_curves.png')
    plot_confusion_matrices(all_models, X_test, y_test, output_dir='reports/figures')
    
    print("\nAnalysis Complete. Check 'reports/' for outputs.")

if __name__ == "__main__":
    main()

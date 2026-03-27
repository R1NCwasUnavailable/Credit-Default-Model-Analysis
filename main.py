import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import get_baseline_models, get_nonlinear_models, get_ensemble_models, get_new_linear_models, get_new_ensemble_models, get_tuning_grids
from src.evaluation import evaluate_model
from src.visualization import plot_roc_curves, plot_confusion_matrices, plot_shap_values

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
    
    # Save processed data
    print("Saving processed data...")
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # 3. Define Models
    baseline = get_baseline_models()
    nonlinear = get_nonlinear_models()
    ensemble = get_ensemble_models(baseline, nonlinear)
    new_linear = get_new_linear_models()
    new_ensemble = get_new_ensemble_models()
    
    tuning_grids = get_tuning_grids()
    
    all_models = {**baseline, **nonlinear, **ensemble, **new_linear, **new_ensemble}
    
    # 4. Train & Evaluate
    results = []
    print("\nTraining & Evaluating Models...")
    
    # Create reports directory
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    for name, model in all_models.items():
        print(f" - Training {name}...")
        
        # Determine if we should subsample for speed
        if name in ['SVM', 'Voting Classifier', 'Stacking Classifier'] and len(X_train) > 5000:
             print(f"   [Optimization] Subsampling training data for {name} to 5000 samples for speed...")
             X_fit, _, y_fit, _ = train_test_split(X_train, y_train, train_size=5000, stratify=y_train, random_state=42)
        else:
             X_fit, y_fit = X_train, y_train
        
        if name in tuning_grids:
             print(f"   [Tuning] Running RandomizedSearchCV for {name}...")
             search = RandomizedSearchCV(
                 estimator=model,
                 param_distributions=tuning_grids[name],
                 n_iter=10,
                 cv=5,
                 scoring='roc_auc',
                 n_jobs=-1,
                 random_state=42
             )
             search.fit(X_fit, y_fit)
             print(f"   Best params: {search.best_params_}")
             model = search.best_estimator_
        else:
             model.fit(X_fit, y_fit)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"   Done. ROC-AUC: {metrics['ROC-AUC']}")

        # Save model reference back to all_models for visualization later
        all_models[name] = model

    # 5. Compile Results
    results_df = pd.DataFrame(results)
    # Sort by ROC-AUC if possible
    try:
        results_df = results_df.sort_values(by='ROC-AUC', ascending=False)
    except:
        pass
    print("\n--- Model Comparison ---")
    print(results_df)
    results_df.to_csv('reports/model_comparison.csv', index=False)
    
    # 6. Visualization
    print("\nGenerating Plots...")
    plot_roc_curves(all_models, X_test, y_test, output_path='reports/roc_curves.png')
    plot_confusion_matrices(all_models, X_test, y_test, output_dir='reports/figures')
    
    # Generate SHAP for the best performing model
    best_model_name = results_df.iloc[0]['Model']
    best_model = all_models[best_model_name]
    print(f"\nGenerating SHAP plots for the best model: {best_model_name}...")
    plot_shap_values(best_model, X_test_df, best_model_name, output_dir='reports/figures')
    
    print("\nAnalysis Complete. Check 'reports/' for outputs.")

if __name__ == "__main__":
    main()

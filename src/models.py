from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_baseline_models(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
        'Gaussian NB': GaussianNB()
    }

def get_nonlinear_models(random_state=42):
    return {
        'k-NN': KNeighborsClassifier(n_neighbors=5), # sensitive to scaling
        'Decision Tree': DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=random_state, class_weight='balanced') # probability for ROC-AUC
    }

def get_ensemble_models(baseline_models, nonlinear_models, random_state=42):
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_estimators=100)
    gb = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
    
    # Voting Classifier
    # We combine LR, DT, and SVM (example)
    estimators = [
        ('lr', baseline_models['Logistic Regression']),
        ('dt', nonlinear_models['Decision Tree']),
        ('svm', nonlinear_models['SVM'])
    ]
    voting = VotingClassifier(estimators=estimators, voting='soft')
    
    return {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Voting Classifier': voting
    }

def get_new_linear_models(random_state=42):
    return {
        'Ridge Classifier': RidgeClassifier(class_weight='balanced'),
        'SGD Classifier': SGDClassifier(random_state=random_state, class_weight='balanced', max_iter=1000),
        'LDA': LinearDiscriminantAnalysis()
    }

def get_new_ensemble_models(random_state=42):
    # Stacking Classifier uses a baseline as final estimator
    estimators = [
        ('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced', n_estimators=50)),
        ('xgb', xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'))
    ]
    return {
        'AdaBoost': AdaBoostClassifier(random_state=random_state, n_estimators=100),
        'Extra Trees': ExtraTreesClassifier(random_state=random_state, class_weight='balanced', n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=random_state, class_weight='balanced'),
        'CatBoost': cb.CatBoostClassifier(random_state=random_state, verbose=0, auto_class_weights='Balanced'),
        'Stacking Classifier': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'))
    }

def get_tuning_grids():
    return {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        },
        'CatBoost': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8]
        }
    }

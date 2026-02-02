from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

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

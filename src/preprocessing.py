import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Cleans the dataset:
    - Renames PAY_0 to PAY_1
    - Regroups Education and Marriage categories
    """
    # Copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Rename PAY_0 if exists
    if 'PAY_0' in df.columns:
        df = df.rename(columns={'PAY_0': 'PAY_1'})
        
    # Education: 0, 5, 6 -> 4 (Other)
    # 1=Graduate, 2=University, 3=High School, 4=Others
    fill = (df.EDUCATION == 0) | (df.EDUCATION == 5) | (df.EDUCATION == 6)
    df.loc[fill, 'EDUCATION'] = 4
    
    # Marriage: 0 -> 3 (Other)
    # 1=Married, 2=Single, 3=Others
    fill = (df.MARRIAGE == 0)
    df.loc[fill, 'MARRIAGE'] = 3
    
    return df

def preprocess_data(df, target='default payment next month', test_size=0.2, random_state=42):
    """
    Splits and scales the data.
    """
    # 1. Clean
    df_clean = clean_data(df)
    
    # 1.5 One-Hot Encoding
    df_clean = pd.get_dummies(df_clean, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
    
    # 2. X, y
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # 4. Scale
    # Identify numerical columns for scaling
    # We will scale everything except the categorical indices (SEX, EDUCATION, MARRIAGE, PAY_X)?
    # Actually PAY_X are ordinal/categorical but often treated as numerical. 
    # For distance based models (KNN, SVM), scaling PAY_X (-2 to 8) is good.
    # EDUCATION and MARRIAGE and SEX are categorical.
    # However, keeping them as integers for tree models is fine. 
    # For One-Hot Encoding, we might want to separate them.
    # For now, we will scale the continuous variables: LIMIT_BAL, AGE, BILL_*, PAY_AMT*
    
    scale_cols = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # 5. Handle Imbalance (SMOTE on training data only)
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)
        print("Target balance (Train):")
        print(y_train.value_counts(normalize=True))

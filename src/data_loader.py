import pandas as pd
import os

def load_data(filepath='data/raw/default of credit card clients.xls'):
    """
    Loads the dataset.
    Note: The original file has a header on the second row (index 1).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read excel, skipping the first row which is often just descriptions in this dataset
    # We need to verify if header is on row 0 or 1.
    # Usually UCI I-Cheng Yeh dataset has header on row 1 (0-indexed).
    try:
        df = pd.read_excel(filepath, header=1)
        
        # Renaissance of the column ID to something more standard if needed, or drop it
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
            
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("\nMissing values:")
        print(df.isnull().sum().sum())
        print("\nTarget distribution (default payment next month):")
        print(df['default payment next month'].value_counts(normalize=True))

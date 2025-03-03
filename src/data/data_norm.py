import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def normalize_data():
    """Normalize data after splitting"""
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Load split data
    print("Loading split data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # Identify datetime columns and non-numeric columns
    datetime_columns = []
    for col in X_train.columns:
        # Check if column has non-numeric data
        if X_train[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_train[col]):
            # Try to parse as datetime
            try:
                pd.to_datetime(X_train[col].iloc[0])
                datetime_columns.append(col)
                print(f"Detected datetime column: {col}")
            except (ValueError, TypeError):
                # If not datetime but still non-numeric, also exclude
                datetime_columns.append(col)
                print(f"Detected non-numeric column: {col}")
    
    # Separate datetime/non-numeric and numeric columns
    X_train_non_numeric = X_train[datetime_columns].copy() if datetime_columns else pd.DataFrame()
    X_test_non_numeric = X_test[datetime_columns].copy() if datetime_columns else pd.DataFrame()
    
    # Drop non-numeric columns for normalization
    X_train_numeric = X_train.drop(columns=datetime_columns, errors='ignore')
    X_test_numeric = X_test.drop(columns=datetime_columns, errors='ignore')
    
    # Normalize numeric data
    print("Normalizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    
    # Convert back to DataFrame with column names
    X_train_numeric_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_test_numeric_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)
    
    # Recombine with non-numeric columns if they exist
    if not X_train_non_numeric.empty:
        X_train_final = pd.concat([X_train_non_numeric, X_train_numeric_scaled], axis=1)
        X_test_final = pd.concat([X_test_non_numeric, X_test_numeric_scaled], axis=1)
    else:
        X_train_final = X_train_numeric_scaled
        X_test_final = X_test_numeric_scaled
    
    # Preserve original column order
    X_train_final = X_train_final[X_train.columns]
    X_test_final = X_test_final[X_test.columns]
    
    # Save normalized data
    print("Saving normalized data...")
    X_train_final.to_csv('data/processed/X_train_normalized.csv', index=False)
    X_test_final.to_csv('data/processed/X_test_normalized.csv', index=False)
    
    print("Normalization completed successfully.")

if __name__ == '__main__':
    normalize_data()
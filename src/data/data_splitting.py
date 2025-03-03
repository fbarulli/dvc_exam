import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path):
    """Split data into training and testing sets"""
    
    os.makedirs('data/processed', exist_ok=True)

    
    print("Loading data...")
    df = pd.read_csv(input_path)

    
    if 'date' in df.columns:
        print("Converting 'date' column to datetime...")
        df['date'] = pd.to_datetime(df['date'])

    
    
    print("Preparing features and target...")
    X = df.drop(columns=['silica_concentrate', 'date'])  
    y = df['silica_concentrate']  

    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    
    print("Saving split datasets...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("\nData splitting completed successfully:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

if __name__ == '__main__':
    
    input_path = '../../data/raw_data/raw.csv'
    split_data(input_path)
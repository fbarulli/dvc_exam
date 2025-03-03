import pandas as pd
import pickle
import json
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    """Evaluate the Ridge model, make predictions, and save results"""

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)


    print("Loading processed datasets...")
    X_train_normalized = pd.read_csv('data/processed/X_train_normalized.csv')
    X_test_normalized = pd.read_csv('data/processed/X_test_normalized.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')


    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    print("Loading best parameters...")
    with open('models/best_ridge_params.pkl', 'rb') as f:
        best_params = pickle.load(f)

 
    print("Training Ridge model with best parameters...")
    ridge_model = Ridge(**best_params, random_state=42)
    ridge_model.fit(X_train_normalized, y_train)


    print("Making predictions...")
    y_pred = ridge_model.predict(X_test_normalized)


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    predictions_df = pd.DataFrame({
        'actual_silica_concentrate': y_test,
        'predicted_silica_concentrate': y_pred
    })
    predictions_path = 'data/processed/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)


    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }
    metrics_path = 'metrics/scores.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\nModel evaluation completed successfully:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == '__main__':
    evaluate_model()
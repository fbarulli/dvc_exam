import pandas as pd
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

def grid_search_ridge():
    """Perform GridSearchCV for Ridge Regression and save best parameters"""
    
    os.makedirs('models', exist_ok=True)

    
    print("Loading processed datasets...")
    X_train_normalized = pd.read_csv('data/processed/X_train_normalized.csv')
    X_test_normalized = pd.read_csv('data/processed/X_test_normalized.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    
    ridge = Ridge(random_state=42)

    
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']  
    }

    
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,  
        n_jobs=-1,  
        verbose=1
    )

    
    grid_search.fit(X_train_normalized, y_train)

    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  
    test_score = mean_squared_error(y_test, grid_search.predict(X_test_normalized))

    
    output_path = 'models/best_ridge_params.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(best_params, f)


if __name__ == '__main__':
    grid_search_ridge()
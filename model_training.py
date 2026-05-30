import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def split_data(df, target_col, test_size=0.2):
    # Ensure only numerical data is passed to model
    df_numeric = df.select_dtypes(include=['number'])
    if target_col not in df_numeric.columns:
        raise ValueError("Target column must be numerical. Please encode it first.")
    
    X = df_numeric.drop(columns=[target_col])
    y = df_numeric[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_regression_model(X_train, y_train, model_type='Linear Regression', tune=False):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'XGBoost Regressor': XGBRegressor(random_state=42)
    }
    
    model = models.get(model_type, LinearRegression())
    
    if tune and model_type == 'Random Forest Regressor':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, predictions

def save_model(model, filename="trained_model.pkl"):
    joblib.dump(model, filename)

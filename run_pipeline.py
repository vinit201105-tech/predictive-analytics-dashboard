import pandas as pd
import data_preprocessing as dp
import model_training as mt
import forecasting as fc
import report_generator as rg
import os

print("Starting end-to-end automated pipeline...")

# 1. Load Data
try:
    df = pd.read_csv('sample_dataset.csv')
    print(f"Loaded dataset successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("sample_dataset.csv not found. Please run create_sample_data.py first.")
    exit(1)

# 2. Preprocess Data
print("Preprocessing data...")
df = dp.handle_missing_values(df, strategy='mean')
df = dp.remove_duplicates(df)

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
df = dp.handle_outliers(df, numeric_cols)

df, encoders = dp.encode_categorical(df)
print(f"Data cleaned and encoded. New shape: {df.shape}")

# 3. Model Training
print("Training Regression Model (Random Forest Regressor)...")
target_col = 'Sales'
X_train, X_test, y_train, y_test = mt.split_data(df, target_col)
model = mt.train_regression_model(X_train, y_train, model_type='Random Forest Regressor', tune=False)
metrics, predictions = mt.evaluate_model(model, X_test, y_test)
print(f"Regression Metrics: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")

mt.save_model(model, 'trained_model.pkl')
print("Model saved as 'trained_model.pkl'")

# 4. Forecasting
print("Running Time-Series Forecasting (Prophet)...")
# We need the original dates for prophet, so we reload and clean just the missing values
df_forecast = pd.read_csv('sample_dataset.csv')
df_forecast = dp.handle_missing_values(df_forecast, strategy='mean')
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'], errors='coerce')
df_forecast = df_forecast.dropna(subset=['Date'])

ts_metrics, ts_preds, forecast_df = fc.train_evaluate_prophet(df_forecast, 'Date', 'Sales', periods=30)
print(f"Forecasting completed. Forecast shape: {forecast_df.shape}")

# 5. Report Generation
print("Generating PDF Report...")
summary = f"Automated Pipeline Run.\nCleaned dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\nModel successfully captured trends."
rg.generate_pdf_report(summary, 'Random Forest Regressor', metrics, filename='automated_report.pdf')
print("Report generated as 'automated_report.pdf'")

print("All tasks completed successfully!")

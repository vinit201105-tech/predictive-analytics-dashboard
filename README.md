# Predictive Analytics Using Historical Data

An end-to-end machine learning application built with Python and Streamlit that allows users to upload historical datasets, clean the data, train predictive models, forecast future trends, evaluate model performance, and visualize results through an interactive dashboard.

## Features

- **Data Upload**: Upload CSV and Excel files.
- **Data Preprocessing**: Handle missing values, remove duplicates, handle outliers, encode categorical variables, and scale numerical features.
- **Exploratory Data Analysis (EDA)**: Interactive visualizations including correlation heatmaps, histograms, box plots, and scatter plots using Plotly.
- **Prediction Models**: 
  - Regression: Linear Regression, Random Forest, Decision Tree, XGBoost.
  - Time Series: ARIMA, Prophet, Moving Average.
- **Model Training**: Train-test split, hyperparameter tuning, model evaluation (MAE, MSE, RMSE, R²).
- **Forecasting**: Predict future values with user-specified horizons.
- **Export**: Download trained models and generate PDF reports.

## Technology Stack

- Frontend: Streamlit, Plotly
- Backend: Python
- Libraries: Pandas, NumPy, Scikit-Learn, Statsmodels, Prophet, XGBoost, FPDF, Joblib

## Installation

1. Clone the repository
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Sample Data

A script to generate a sample dataset is provided. Run `python create_sample_data.py` to create `sample_dataset.csv`, which can be used to test the application.

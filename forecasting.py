import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_evaluate_arima(series, order=(1,1,1), steps=30):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    predictions = fitted_model.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    
    # Refit on full data for future forecast
    full_model = ARIMA(series, order=order).fit()
    future_forecast = full_model.forecast(steps=steps)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, predictions, future_forecast

def train_evaluate_prophet(df, date_col, target_col, periods=30):
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
    
    train_size = int(len(prophet_df) * 0.8)
    train, test = prophet_df.iloc[:train_size], prophet_df.iloc[train_size:]
    
    m = Prophet(daily_seasonality=True)
    m.fit(train)
    
    future = m.make_future_dataframe(periods=len(test))
    forecast = m.predict(future)
    
    predictions = forecast['yhat'].iloc[train_size:].values
    test_y = test['y'].values
    
    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mape = mean_absolute_percentage_error(test_y, predictions)
    
    # Real future forecast
    m_full = Prophet(daily_seasonality=True)
    m_full.fit(prophet_df)
    future_full = m_full.make_future_dataframe(periods=periods)
    forecast_full = m_full.predict(future_full)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, predictions, forecast_full

def train_evaluate_moving_average(series, window=3, steps=30):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Calculate moving average predictions for test set
    full_series = pd.Series(series)
    rolling_mean = full_series.rolling(window=window).mean()
    predictions = rolling_mean.iloc[train_size:train_size+len(test)].bfill().values
    
    if len(predictions) > len(test):
        predictions = predictions[:len(test)]
    elif len(predictions) < len(test):
        predictions = np.append(predictions, [predictions[-1]] * (len(test) - len(predictions)))
        
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    
    # Future forecast
    future_forecast = []
    current_series = list(series)
    for _ in range(steps):
        next_val = np.mean(current_series[-window:])
        future_forecast.append(next_val)
        current_series.append(next_val)
        
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, predictions, future_forecast

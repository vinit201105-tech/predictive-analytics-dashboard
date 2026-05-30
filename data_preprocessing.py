import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def handle_missing_values(df, strategy='mean'):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                mode_vals = df_clean[col].mode()
                if not mode_vals.empty:
                    df_clean[col].fillna(mode_vals[0], inplace=True)
            else:
                if strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategy == 'drop':
                    df_clean.dropna(subset=[col], inplace=True)
    return df_clean

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    return df_clean

def encode_categorical(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    encoders = {}
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    return df_encoded, encoders

def scale_features(df, target_col=None, method='standard'):
    df_scaled = df.copy()
    features = [col for col in df.columns if col != target_col]
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    if len(features) > 0:
        # Only scale numerical features
        num_features = df_scaled[features].select_dtypes(include=['number']).columns.tolist()
        if num_features:
            df_scaled[num_features] = scaler.fit_transform(df_scaled[num_features])
    return df_scaled, scaler

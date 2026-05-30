import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    data = {
        'Date': dates,
        'Sales': np.random.normal(500, 50, 365) + np.linspace(0, 200, 365),
        'Customers': np.random.normal(100, 15, 365).astype(int),
        'Marketing_Spend': np.random.normal(200, 30, 365),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food'], 365)
    }
    df = pd.DataFrame(data)
    # Add some realistic flaws
    df.loc[10:15, 'Sales'] = np.nan # Add missing values
    df = pd.concat([df, df.iloc[[5, 20]]], ignore_index=True) # Add duplicates
    
    df.to_csv('sample_dataset.csv', index=False)
    print("Sample dataset 'sample_dataset.csv' generated successfully!")

if __name__ == "__main__":
    create_sample_data()

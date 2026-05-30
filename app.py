import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import plotly.express as px
from datetime import timedelta

# Import custom modules
import data_preprocessing as dp
import model_training as mt
import forecasting as fc
import visualization as vz
import report_generator as rg

st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for modern UI and Dark mode support
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4A90E2; font-weight: bold; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #50E3C2; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .metric-card { background-color: #1E1E1E; padding: 1rem; border-radius: 8px; border-left: 4px solid #4A90E2; }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    menu = ["Data Upload", "Data Preprocessing", "Exploratory Data Analysis", "Model Training", "Forecasting", "Export & Report"]
    choice = st.sidebar.radio("Go to:", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Predictive Analytics Dashboard. Upload data, process it, train models, and forecast the future.")
    
    st.markdown('<div class="main-header">Predictive Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'clean_data' not in st.session_state:
        st.session_state['clean_data'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = None
        
    if choice == "Data Upload":
        st.markdown('<div class="sub-header">1. Data Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            with st.spinner("Loading dataset..."):
                try:
                    if uploaded_file.name.endswith('csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state['data'] = df
                    st.session_state['clean_data'] = df.copy()
                    st.success("Data uploaded successfully!")
                    
                    st.write("### Dataset Preview")
                    st.dataframe(df, height=400)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Dataset Shape")
                        st.info(f"{df.shape[0]} rows, {df.shape[1]} columns")
                    with col2:
                        st.write("### Data Types")
                        st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Type'}))
                        
                except Exception as e:
                    st.error(f"Error loading data: {e}")

    elif choice == "Data Preprocessing":
        st.markdown('<div class="sub-header">2. Data Preprocessing</div>', unsafe_allow_html=True)
        if st.session_state['data'] is not None:
            df = st.session_state['clean_data']
            
            st.write("### Missing Values")
            missing_data = df.isnull().sum()
            st.write(missing_data[missing_data > 0])
            if missing_data.sum() > 0:
                missing_strategy = st.selectbox("Imputation Strategy", ['mean', 'median', 'drop'])
                if st.button("Handle Missing Values"):
                    df = dp.handle_missing_values(df, strategy=missing_strategy)
                    st.session_state['clean_data'] = df
                    st.success("Missing values handled.")
                    st.rerun()
            else:
                st.success("No missing values found.")
                
            st.markdown("---")
            st.write("### Remove Duplicates")
            dup_count = df.duplicated().sum()
            st.write(f"Found {dup_count} duplicate rows.")
            if dup_count > 0 and st.button("Remove Duplicates"):
                df = dp.remove_duplicates(df)
                st.session_state['clean_data'] = df
                st.success("Duplicates removed.")
                st.rerun()
                
            st.markdown("---")
            st.write("### Handle Outliers")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                outlier_cols = st.multiselect("Select columns for outlier treatment (IQR method)", numeric_cols)
                if st.button("Handle Outliers"):
                    df = dp.handle_outliers(df, outlier_cols)
                    st.session_state['clean_data'] = df
                    st.success("Outliers handled.")
            
            st.markdown("---")
            st.write("### Encode Categorical Variables")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.write(f"Categorical columns found: {', '.join(cat_cols) if cat_cols else 'None'}")
            if cat_cols and st.button("Encode Categorical Variables"):
                df, _ = dp.encode_categorical(df)
                st.session_state['clean_data'] = df
                st.success("Categorical variables encoded.")
                st.rerun()
                
            st.markdown("---")
            st.write("### Scale Features")
            scale_method = st.selectbox("Scaling Method", ['standard', 'minmax'])
            target_col_scale = st.selectbox("Select Target Column (to exclude from scaling)", ['None'] + df.columns.tolist())
            if st.button("Scale Numerical Features"):
                tgt = None if target_col_scale == 'None' else target_col_scale
                df, _ = dp.scale_features(df, target_col=tgt, method=scale_method)
                st.session_state['clean_data'] = df
                st.success("Features scaled.")
                
            st.markdown("---")
            st.write("### Preprocessed Data Preview")
            st.dataframe(df, height=400)
        else:
            st.warning("Please upload data first.")

    elif choice == "Exploratory Data Analysis":
        st.markdown('<div class="sub-header">3. Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
        if st.session_state['clean_data'] is not None:
            df = st.session_state['clean_data']
            
            st.write("### Summary Statistics")
            st.dataframe(df.describe())
            
            st.write("### Correlation Heatmap")
            if len(df.select_dtypes(include=['number']).columns) > 1:
                st.plotly_chart(vz.plot_correlation_heatmap(df), use_container_width=True)
            else:
                st.write("Not enough numeric columns for correlation.")
                
            st.write("### Distributions & Box Plots")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select Column for Distribution", numeric_cols)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(vz.plot_histogram(df, col), use_container_width=True)
                with col2:
                    st.plotly_chart(vz.plot_boxplot(df, col), use_container_width=True)
                    
            st.write("### Feature Relationships")
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
                with col2:
                    y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)
                st.plotly_chart(vz.plot_scatter(df, x_col, y_col), use_container_width=True)
        else:
            st.warning("Please upload and preprocess data first.")

    elif choice == "Model Training":
        st.markdown('<div class="sub-header">4. Regression Model Training</div>', unsafe_allow_html=True)
        if st.session_state['clean_data'] is not None:
            df = st.session_state['clean_data']
            
            st.info("Ensure all categorical variables are encoded before training regression models.")
            target_col = st.selectbox("Select Target Variable", df.columns.tolist())
            model_type = st.selectbox("Select Model", ['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor', 'XGBoost Regressor'])
            tune = st.checkbox("Hyperparameter Tuning (Random Forest only)")
            
            if st.button("Train Model"):
                with st.spinner('Training model... This might take a while for large datasets.'):
                    try:
                        X_train, X_test, y_train, y_test = mt.split_data(df, target_col)
                        model = mt.train_regression_model(X_train, y_train, model_type, tune)
                        metrics, predictions = mt.evaluate_model(model, X_test, y_test)
                        
                        st.session_state['model'] = model
                        st.session_state['metrics'] = metrics
                        st.session_state['model_name'] = model_type
                        
                        st.success(f"{model_type} trained successfully!")
                        
                        st.write("### Evaluation Metrics")
                        cols = st.columns(4)
                        cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                        cols[1].metric("MSE", f"{metrics['MSE']:.4f}")
                        cols[2].metric("RMSE", f"{metrics['RMSE']:.4f}")
                        cols[3].metric("R² Score", f"{metrics['R2']:.4f}")
                        
                        st.write("### Actual vs Predicted (Test Set)")
                        # Plot first 100 points for better visibility if dataset is large
                        plot_limit = min(100, len(y_test))
                        st.plotly_chart(vz.plot_actual_vs_predicted(y_test.values[:plot_limit], predictions[:plot_limit]), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during training: {e}")
        else:
            st.warning("Please upload and preprocess data first.")

    elif choice == "Forecasting":
        st.markdown('<div class="sub-header">5. Time Series Forecasting</div>', unsafe_allow_html=True)
        if st.session_state['clean_data'] is not None:
            df = st.session_state['clean_data']
            
            ts_model = st.selectbox("Select Forecasting Model", ['ARIMA', 'Prophet', 'Moving Average'])
            
            if ts_model == 'Prophet':
                date_col = st.selectbox("Select Date Column", df.columns.tolist())
                target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1 if len(df.columns)>1 else 0)
            else:
                target_col = st.selectbox("Select Target Column (Time Series Data)", df.select_dtypes(include=['number']).columns.tolist())
                
            horizon = st.number_input("Forecast Horizon (periods)", min_value=1, max_value=365*5, value=30)
            
            if st.button("Generate Forecast"):
                with st.spinner('Generating forecast...'):
                    try:
                        if ts_model == 'ARIMA':
                            series = df[target_col].dropna().values
                            metrics, predictions, future = fc.train_evaluate_arima(series, steps=horizon)
                            
                            st.write("### Metrics (Test Set)")
                            cols = st.columns(3)
                            cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                            cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
                            cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            
                            hist_dates = np.arange(len(series))
                            fut_dates = np.arange(len(series), len(series)+horizon)
                            st.plotly_chart(vz.plot_forecast(hist_dates, series, fut_dates, future, title="ARIMA Forecast"), use_container_width=True)
                            
                        elif ts_model == 'Prophet':
                            if df[date_col].dtype != 'datetime64[ns]':
                                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                                df = df.dropna(subset=[date_col])
                            
                            metrics, predictions, forecast_df = fc.train_evaluate_prophet(df, date_col, target_col, periods=horizon)
                            
                            st.write("### Metrics (Test Set)")
                            cols = st.columns(3)
                            cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                            cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
                            cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            
                            st.write("### Future Forecast Trend")
                            st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon))
                            
                            fig = px.line(forecast_df, x='ds', y='yhat', title="Prophet Forecast", template="plotly_dark")
                            fig.add_scatter(x=df[date_col], y=df[target_col], mode='lines', name='Actual', opacity=0.5)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif ts_model == 'Moving Average':
                            series = df[target_col].dropna().values
                            metrics, predictions, future = fc.train_evaluate_moving_average(series, steps=horizon)
                            
                            st.write("### Metrics (Test Set)")
                            cols = st.columns(3)
                            cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                            cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
                            cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            
                            hist_dates = np.arange(len(series))
                            fut_dates = np.arange(len(series), len(series)+horizon)
                            st.plotly_chart(vz.plot_forecast(hist_dates, series, fut_dates, future, title="Moving Average Forecast"), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error during forecasting: {e}")
        else:
            st.warning("Please upload and preprocess data first.")

    elif choice == "Export & Report":
        st.markdown('<div class="sub-header">6. Export & Report Generation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Download Trained Model")
            if st.session_state['model'] is not None:
                mt.save_model(st.session_state['model'], "trained_model.pkl")
                with open("trained_model.pkl", "rb") as f:
                    st.download_button("Download Model (.pkl)", data=f, file_name="trained_model.pkl", mime="application/octet-stream")
            else:
                st.info("Train a model first to enable model export.")
                
        with col2:
            st.write("### Generate PDF Report")
            if st.session_state['metrics'] is not None:
                if st.button("Generate Report"):
                    summary = f"The dataset contains {st.session_state['clean_data'].shape[0]} records and {st.session_state['clean_data'].shape[1]} features."
                    report_file = rg.generate_pdf_report(summary, st.session_state.get('model_name', 'Custom Model'), st.session_state['metrics'])
                    with open(report_file, "rb") as f:
                        st.download_button("Download PDF Report", data=f, file_name="predictive_analytics_report.pdf", mime="application/pdf")
            else:
                st.info("Train a model first to enable report generation.")
            
if __name__ == '__main__':
    main()

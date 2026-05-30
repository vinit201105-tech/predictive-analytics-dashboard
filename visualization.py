import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_correlation_heatmap(df):
    corr = df.select_dtypes(include=['number']).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
    return fig

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, title=f"Distribution of {column}", marginal="box", color_discrete_sequence=['#4A90E2'])
    return fig

def plot_boxplot(df, column):
    fig = px.box(df, y=column, title=f"Boxplot of {column}", color_discrete_sequence=['#50E3C2'])
    return fig

def plot_scatter(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", opacity=0.7, color_discrete_sequence=['#E94A4A'])
    return fig

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines+markers', name='Actual', line=dict(color='#4A90E2')))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='#E94A4A', dash='dash')))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Value", template="plotly_dark")
    return fig

def plot_forecast(historical_dates, historical_values, future_dates, future_values, title="Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_values, mode='lines', name='Historical', line=dict(color='#4A90E2')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_values, mode='lines', name='Forecast', line=dict(color='#50E3C2', dash='dash')))
    fig.update_layout(title=title, xaxis_title="Time / Date", yaxis_title="Value", template="plotly_dark")
    return fig

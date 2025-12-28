"""
Revenue at risk calculations for customer retention analysis.
"""

import pandas as pd
import numpy as np


def calculate_revenue_at_risk(df, churn_probabilities, revenue_column=None, avg_revenue=None):
    """
    Calculate revenue at risk for each customer based on churn probability.
    
    Args:
        df: DataFrame with customer data
        churn_probabilities: Array of churn probabilities
        revenue_column: Name of revenue column (if exists)
        avg_revenue: Average revenue per customer (if revenue_column not available)
        
    Returns:
        Series: Revenue at risk per customer
    """
    # Determine customer revenue values
    if revenue_column and revenue_column in df.columns:
        customer_revenue = df[revenue_column].fillna(0)
    elif avg_revenue is not None:
        customer_revenue = pd.Series([avg_revenue] * len(df), index=df.index)
    else:
        # Estimate revenue from numeric columns if available
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use mean of numeric columns as proxy for revenue
            customer_revenue = df[numeric_cols].mean(axis=1).abs()
            customer_revenue = customer_revenue.fillna(customer_revenue.mean() if customer_revenue.mean() > 0 else 1000)
        else:
            # Default to 1000 if no revenue information
            customer_revenue = pd.Series([1000] * len(df), index=df.index)
    
    # Revenue at risk = Revenue * Churn Probability
    revenue_at_risk = pd.Series(customer_revenue.values * churn_probabilities, index=df.index)
    
    return revenue_at_risk, customer_revenue


def calculate_customer_value_risk_matrix(df, churn_probabilities, customer_revenue):
    """
    Create value-risk matrix for customers.
    
    Args:
        df: DataFrame
        churn_probabilities: Array of churn probabilities
        customer_revenue: Series of customer revenue values
        
    Returns:
        DataFrame: Customer data with value_risk_segment column
    """
    result_df = df.copy()
    result_df['churn_probability'] = churn_probabilities
    result_df['customer_revenue'] = customer_revenue
    
    # Define thresholds (median splits)
    revenue_threshold = customer_revenue.median()
    prob_threshold = np.median(churn_probabilities)
    
    # Create segments
    def assign_segment(row):
        high_value = row['customer_revenue'] >= revenue_threshold
        high_risk = row['churn_probability'] >= prob_threshold
        
        if high_value and high_risk:
            return "High Value, High Risk"
        elif high_value and not high_risk:
            return "High Value, Low Risk"
        elif not high_value and high_risk:
            return "Low Value, High Risk"
        else:
            return "Low Value, Low Risk"
    
    result_df['value_risk_segment'] = result_df.apply(assign_segment, axis=1)
    
    return result_df


def create_value_risk_scatter_plot(df_with_segments):
    """
    Create scatter plot of customer value vs churn risk.
    
    Args:
        df_with_segments: DataFrame with churn_probability, customer_revenue, value_risk_segment
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    segment_colors = {
        "High Value, High Risk": "#E24A4A",
        "High Value, Low Risk": "#4A90E2",
        "Low Value, High Risk": "#F5A623",
        "Low Value, Low Risk": "#7ED321"
    }
    
    fig = go.Figure()
    
    for segment in df_with_segments['value_risk_segment'].unique():
        segment_data = df_with_segments[df_with_segments['value_risk_segment'] == segment]
        fig.add_trace(go.Scatter(
            x=segment_data['churn_probability'],
            y=segment_data['customer_revenue'],
            mode='markers',
            name=segment,
            marker=dict(
                color=segment_colors.get(segment, '#808080'),
                size=5,
                opacity=0.6
            )
        ))
    
    fig.update_layout(
        title="Customer Value vs Churn Risk",
        xaxis_title="Churn Probability",
        yaxis_title="Customer Revenue",
        font=dict(size=12),
        height=450,
        hovermode='closest'
    )
    
    return fig


def get_segment_summary(df_with_segments):
    """
    Get summary statistics by value-risk segment.
    
    Args:
        df_with_segments: DataFrame with value_risk_segment column
        
    Returns:
        DataFrame: Summary statistics
    """
    summary = df_with_segments.groupby('value_risk_segment').agg({
        'customer_revenue': ['count', 'mean', 'sum'],
        'churn_probability': 'mean',
    }).round(2)
    
    summary.columns = ['Count', 'Avg Revenue', 'Total Revenue', 'Avg Churn Risk']
    summary = summary.sort_values('Total Revenue', ascending=False)
    
    return summary



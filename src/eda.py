"""
Exploratory Data Analysis and Customer Segmentation utilities.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def compute_basic_stats(df, churn_column, numeric_features):
    """
    Compute basic statistics for the dataset.
    
    Args:
        df: DataFrame
        churn_column: Name of churn column
        numeric_features: List of numeric feature names
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_customers': len(df),
        'churn_rate': df[churn_column].mean(),
        'churn_count': df[churn_column].sum(),
        'retention_count': (1 - df[churn_column]).sum(),
        'numeric_features': len(numeric_features),
    }
    
    if numeric_features:
        stats['numeric_summary'] = df[numeric_features].describe().to_dict()
    
    return stats


def segment_customers(df, numeric_features, n_segments=4):
    """
    Perform customer segmentation using K-means on numeric features.
    
    Args:
        df: DataFrame
        numeric_features: List of numeric feature names
        n_segments: Number of segments to create
        
    Returns:
        tuple: (df_with_segments, segment_stats)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    df_seg = df.copy()
    
    if not numeric_features or len(numeric_features) < 2:
        # Fallback: create simple segments based on available data
        df_seg['segment'] = 'Segment_1'
        segment_stats = {'Segment_1': {'count': len(df_seg), 'features': {}}}
        return df_seg, segment_stats
    
    # Prepare data for clustering
    X_numeric = df[numeric_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=min(n_segments, len(df)), random_state=42, n_init=10)
    df_seg['segment'] = kmeans.fit_predict(X_scaled)
    df_seg['segment'] = 'Segment_' + (df_seg['segment'] + 1).astype(str)
    
    # Compute segment statistics
    segment_stats = {}
    for segment in df_seg['segment'].unique():
        seg_data = df_seg[df_seg['segment'] == segment]
        segment_stats[segment] = {
            'count': len(seg_data),
            'percentage': len(seg_data) / len(df_seg) * 100,
            'features': seg_data[numeric_features].mean().to_dict() if numeric_features else {}
        }
    
    return df_seg, segment_stats


def analyze_churn_drivers(df, churn_column, numeric_features, categorical_features):
    """
    Analyze which features are most associated with churn.
    
    Args:
        df: DataFrame
        churn_column: Name of churn column
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        dict: Analysis results with correlations and associations
    """
    drivers = {
        'numeric_correlations': {},
        'categorical_associations': {}
    }
    
    # Numeric feature correlations with churn
    for feature in numeric_features:
        if feature in df.columns:
            corr = df[[feature, churn_column]].corr().iloc[0, 1]
            if not np.isnan(corr):
                drivers['numeric_correlations'][feature] = corr
    
    # Categorical feature associations (churn rate by category)
    for feature in categorical_features:
        if feature in df.columns:
            churn_rates = df.groupby(feature)[churn_column].agg(['mean', 'count'])
            churn_rates = churn_rates[churn_rates['count'] >= 5]  # Filter small groups
            if len(churn_rates) > 0:
                drivers['categorical_associations'][feature] = churn_rates.to_dict('index')
    
    # Sort by absolute correlation/association strength
    drivers['numeric_correlations'] = dict(
        sorted(drivers['numeric_correlations'].items(), 
               key=lambda x: abs(x[1]), reverse=True)
    )
    
    return drivers


def create_churn_distribution_plot(df, churn_column):
    """
    Create churn distribution visualization.
    
    Args:
        df: DataFrame
        churn_column: Name of churn column
        
    Returns:
        plotly figure
    """
    churn_counts = df[churn_column].value_counts()
    labels = ['Retained', 'Churned']
    values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=['#4A90E2', '#E24A4A']
    )])
    
    fig.update_layout(
        title="Customer Churn Distribution",
        showlegend=True,
        font=dict(size=12),
        height=350
    )
    
    return fig


def create_segment_churn_plot(df_seg, churn_column):
    """
    Create churn rate by segment visualization.
    
    Args:
        df_seg: DataFrame with segment column
        churn_column: Name of churn column
        
    Returns:
        plotly figure
    """
    segment_churn = df_seg.groupby('segment')[churn_column].agg(['mean', 'count'])
    segment_churn = segment_churn.sort_values('mean', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=segment_churn.index,
            y=segment_churn['mean'],
            text=[f"{val:.1%}" for val in segment_churn['mean']],
            textposition='auto',
            marker_color='#4A90E2'
        )
    ])
    
    fig.update_layout(
        title="Churn Rate by Customer Segment",
        xaxis_title="Segment",
        yaxis_title="Churn Rate",
        yaxis=dict(tickformat='.1%'),
        font=dict(size=12),
        height=350
    )
    
    return fig


def create_feature_correlation_plot(correlations_dict, top_n=10):
    """
    Create feature correlation with churn visualization.
    
    Args:
        correlations_dict: Dictionary of feature names to correlation values
        top_n: Number of top features to show
        
    Returns:
        plotly figure
    """
    # Get top N features by absolute correlation
    sorted_features = sorted(correlations_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features = [f[0] for f in top_features]
    correlations = [f[1] for f in top_features]
    
    colors = ['#4A90E2' if c > 0 else '#E24A4A' for c in correlations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=correlations,
            y=features,
            orientation='h',
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Features Correlated with Churn",
        xaxis_title="Correlation with Churn",
        yaxis_title="Feature",
        font=dict(size=12),
        height=400
    )
    
    return fig



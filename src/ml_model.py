"""
CatBoost-based churn risk modeling with safe tuning.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib


def prepare_categorical_indices(X, categorical_features):
    """
    Get indices of categorical features for CatBoost.
    
    Args:
        X: DataFrame with features
        categorical_features: List of categorical feature names
        
    Returns:
        list: Indices of categorical columns
    """
    cat_indices = []
    for i, col in enumerate(X.columns):
        if col in categorical_features:
            cat_indices.append(i)
    return cat_indices


def train_churn_model(X, y, categorical_features, validation_size=0.2):
    """
    Train CatBoost classifier with safe, conservative tuning.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        categorical_features: List of categorical feature names
        validation_size: Proportion of data to use for validation
        
    Returns:
        tuple: (trained_model, train_metrics, val_metrics, feature_importance)
    """
    # Prepare categorical indices
    cat_indices = prepare_categorical_indices(X, categorical_features)
    
    # Split data with stratification to preserve churn distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=42, stratify=y
    )
    
    # Conservative CatBoost parameters for stability and calibration
    model_params = {
        'iterations': 200,  # Moderate number of iterations
        'learning_rate': 0.05,  # Lower learning rate for stability
        'depth': 4,  # Shallow trees to prevent overfitting
        'l2_leaf_reg': 10,  # Strong L2 regularization
        'border_count': 32,  # Reduced border count for faster training
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 20,  # Early stopping to prevent overfitting
        'use_best_model': True,
        'cat_features': cat_indices if cat_indices else None,
    }
    
    # Initialize model
    model = CatBoostClassifier(**model_params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
    }
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
    }
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, train_metrics, val_metrics, feature_importance


def predict_churn_risk(model, X):
    """
    Predict churn probabilities for customers.
    
    Args:
        model: Trained CatBoost model
        X: Feature DataFrame
        
    Returns:
        numpy array: Churn probabilities
    """
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities


def create_probability_distribution_plot(probabilities):
    """
    Create histogram of churn probabilities.
    
    Args:
        probabilities: Array of churn probabilities
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Histogram(
            x=probabilities,
            nbinsx=30,
            marker_color='#4A90E2',
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title="Distribution of Churn Probabilities",
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
        font=dict(size=12),
        height=350
    )
    
    return fig


def create_feature_importance_plot(feature_importance, top_n=15):
    """
    Create feature importance visualization.
    
    Args:
        feature_importance: DataFrame with feature and importance columns
        top_n: Number of top features to show
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#4A90E2'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        font=dict(size=12),
        height=400
    )
    
    return fig



"""
Data utilities for loading, validating, and cleaning customer churn data.
"""

import pandas as pd
import numpy as np


def load_and_validate_data(uploaded_file):
    """
    Load CSV file and perform validation checks.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (DataFrame, validation_errors)
    """
    if uploaded_file is None:
        return None, ["No file uploaded"]
    
    validation_errors = []
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        validation_errors.append(f"Error reading CSV: {str(e)}")
        return None, validation_errors
    
    # Basic structure validation
    if df.empty:
        validation_errors.append("File is empty")
        return None, validation_errors
    
    if len(df) < 10:
        validation_errors.append(f"Insufficient data: {len(df)} rows (minimum 10 required)")
    
    # Check for required columns (flexible - will infer churn column)
    if len(df.columns) < 3:
        validation_errors.append("File must have at least 3 columns")
    
    return df, validation_errors


def detect_churn_column(df):
    """
    Automatically detect the churn column in the dataset.
    
    Args:
        df: DataFrame
        
    Returns:
        str: Name of churn column, or None if not found
    """
    # Look for common churn column names (case-insensitive)
    churn_keywords = ['churn', 'target', 'is_churn', 'churned', 'attrition']
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in churn_keywords:
            if keyword in col_lower:
                return col
    
    # Check if any column is binary (0/1 or True/False) and might be churn
    for col in df.columns:
        if df[col].dtype in ['int64', 'bool']:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                return col
    
    return None


def clean_data(df, churn_column=None):
    """
    Clean and preprocess the dataset.
    
    Args:
        df: DataFrame
        churn_column: Name of churn column (auto-detected if None)
        
    Returns:
        tuple: (cleaned_df, churn_column, cleaning_info)
    """
    df = df.copy()
    cleaning_info = []
    
    # Detect churn column if not provided
    if churn_column is None:
        churn_column = detect_churn_column(df)
        if churn_column:
            cleaning_info.append(f"Auto-detected churn column: {churn_column}")
        else:
            cleaning_info.append("Warning: Could not auto-detect churn column")
    
    # Convert churn column to binary 0/1
    if churn_column and churn_column in df.columns:
        if df[churn_column].dtype == bool:
            df[churn_column] = df[churn_column].astype(int)
        elif df[churn_column].dtype == 'object':
            # Try to convert string values to binary
            unique_vals = df[churn_column].dropna().unique()
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df[churn_column] = df[churn_column].map(mapping)
    
    # Handle missing values
    initial_missing = df.isnull().sum().sum()
    if initial_missing > 0:
        # Drop rows with missing churn target
        if churn_column and churn_column in df.columns:
            df = df.dropna(subset=[churn_column])
            cleaning_info.append(f"Dropped rows with missing churn values")
        
        # For other columns, fill numeric with median, categorical with mode
        for col in df.columns:
            if col != churn_column:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        final_missing = df.isnull().sum().sum()
        cleaning_info.append(f"Handled missing values: {initial_missing} → {final_missing}")
    
    # Remove duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_len - len(df)
    if duplicates_removed > 0:
        cleaning_info.append(f"Removed {duplicates_removed} duplicate rows")
    
    # Ensure numeric columns are properly typed
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df, churn_column, cleaning_info


def prepare_features(df, churn_column):
    """
    Separate features and target, and identify column types.
    
    Args:
        df: DataFrame
        churn_column: Name of churn column
        
    Returns:
        tuple: (X, y, categorical_features, numeric_features)
    """
    if churn_column is None or churn_column not in df.columns:
        raise ValueError("Churn column not found in dataframe")
    
    X = df.drop(columns=[churn_column])
    y = df[churn_column]
    
    # Identify categorical and numeric features
    categorical_features = []
    numeric_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_features.append(col)
        elif X[col].dtype in ['int64', 'float64']:
            # Check if numeric column has few unique values (might be categorical)
            unique_count = X[col].nunique()
            if unique_count <= 10 and unique_count < len(X) * 0.1:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            numeric_features.append(col)
    
    return X, y, categorical_features, numeric_features


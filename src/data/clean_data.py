"""Data cleaning module for blitz model"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_blitz_data(df: pd.DataFrame, target_col: str = "blitz") -> pd.DataFrame:
    """
    Clean and preprocess blitz model data.
    
    Args:
        df: Raw DataFrame with blitz features
        target_col: Name of target column
        
    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    logger.info(f"Starting data cleaning. Shape: {df.shape}")
    
    # Create a copy
    clean_df = df.copy()
    
    # Remove rows with missing target
    initial_rows = len(clean_df)
    clean_df = clean_df.dropna(subset=[target_col])
    removed_target = initial_rows - len(clean_df)
    logger.info(f"Removed {removed_target} rows with missing target")
    
    # Remove rows where all feature columns are null
    feature_cols = [col for col in clean_df.columns if col != target_col]
    clean_df = clean_df.dropna(subset=feature_cols, how="all")
    removed_all_null = initial_rows - removed_target - len(clean_df)
    logger.info(f"Removed {removed_all_null} rows with all null features")
    
    # Fill missing values for categorical columns with 'Unknown'
    categorical_cols = [
        "offense_personnel",
        "defense_personnel",
        "formation",
    ]
    for col in categorical_cols:
        if col in clean_df.columns:
            clean_df[col].fillna("Unknown", inplace=True)
    
    # Fill missing values for numeric columns with median
    numeric_cols = [
        "down",
        "ydstogo",
        "yardline_100",
        "quarter",
        "game_seconds_remaining",
        "score_differential",
    ]
    for col in numeric_cols:
        if col in clean_df.columns and clean_df[col].dtype in ["float64", "int64"]:
            clean_df[col].fillna(clean_df[col].median(), inplace=True)
    
    # Fill boolean columns
    bool_cols = ["shotgun", "motion"]
    for col in bool_cols:
        if col in clean_df.columns:
            clean_df[col].fillna(0, inplace=True)
    
    logger.info(f"Cleaned data shape: {clean_df.shape}")
    logger.info(f"Missing values:\n{clean_df.isnull().sum()}")
    
    return clean_df


def validate_blitz_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that cleaned data has required columns and structure.
    
    Args:
        df: Cleaned DataFrame
        required_columns: List of required columns
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for null values in required columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Remaining nulls in required columns:\n{null_counts[null_counts > 0]}")
    
    logger.info("Data validation passed")
    return True


def get_class_distribution(df: pd.DataFrame, target_col: str = "blitz") -> dict:
    """
    Get class distribution of target variable.
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        
    Returns:
        Dictionary with class counts and percentages
    """
    distribution = df[target_col].value_counts().to_dict()
    total = len(df)
    percentages = {k: (v / total) * 100 for k, v in distribution.items()}
    
    logger.info(f"Class distribution (counts): {distribution}")
    logger.info(f"Class distribution (%):\n  0 (No Blitz): {percentages.get(0, 0):.2f}%")
    logger.info(f"  1 (Blitz): {percentages.get(1, 0):.2f}%")
    
    return {"counts": distribution, "percentages": percentages}


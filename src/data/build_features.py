"""Feature engineering module for blitz model"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: list,
    fit_encoders: bool = True,
    encoders: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame with categorical features
        categorical_cols: List of categorical column names
        fit_encoders: Whether to fit new encoders or use existing ones
        encoders: Dictionary of existing encoders to use
        
    Returns:
        Tuple of (encoded_df, encoders_dict)
    """
    df_encoded = df.copy()
    encoders_dict = encoders if encoders is not None else {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        if fit_encoders:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df[col].astype(str))
            encoders_dict[col] = encoder
            logger.info(f"Encoded {col}: {len(encoder.classes_)} unique values")
        else:
            if col not in encoders_dict:
                raise ValueError(f"Encoder for {col} not found")
            df_encoded[col] = encoders_dict[col].transform(df[col].astype(str))
    
    return df_encoded, encoders_dict


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from base features.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with additional derived features
    """
    df_features = df.copy()
    
    # Time-based features
    if "quarter" in df.columns and "game_seconds_remaining" in df.columns:
        # Game progress (0-1)
        df_features["game_progress"] = (4 - df["quarter"]) / 4
        
        # Seconds elapsed (normalized)
        total_seconds = 3600  # 60 min * 60 sec
        df_features["seconds_elapsed"] = (total_seconds - df["game_seconds_remaining"]) / total_seconds
    
    # Down-based features
    if "down" in df.columns and "ydstogo" in df.columns:
        # Down-distance interaction
        df_features["down_ydstogo_interaction"] = df["down"] * df["ydstogo"]
        
        # Is it a crucial down (3rd or 4th)
        df_features["critical_down"] = ((df["down"] == 3) | (df["down"] == 4)).astype(int)
    
    # Field position features
    if "yardline_100" in df.columns:
        # Red zone (within 20 yards)
        df_features["in_redzone"] = (df["yardline_100"] <= 20).astype(int)
        
        # Goal line (within 5 yards)
        df_features["near_goalline"] = (df["yardline_100"] <= 5).astype(int)
        
        # Own territory
        df_features["own_territory"] = (df["yardline_100"] > 50).astype(int)
    
    logger.info(f"Created {len(df_features.columns) - len(df.columns)} derived features")
    logger.info(f"Total features: {len(df_features.columns)}")
    
    return df_features


def normalize_numeric_features(
    df: pd.DataFrame,
    numeric_cols: list,
    normalize_params: dict = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize numeric features to [0, 1] range using min-max scaling.
    
    Args:
        df: DataFrame with numeric features
        numeric_cols: List of numeric column names
        normalize_params: Dictionary with min/max values for each column
        fit: Whether to fit new parameters or use existing ones
        
    Returns:
        Tuple of (normalized_df, params_dict)
    """
    df_normalized = df.copy()
    params = normalize_params if normalize_params is not None else {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        if fit:
            col_min = df[col].min()
            col_max = df[col].max()
            params[col] = {"min": col_min, "max": col_max}
            
            if col_max - col_min > 0:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
                
            logger.info(f"Normalized {col}: [{col_min}, {col_max}]")
        else:
            if col not in params:
                raise ValueError(f"Normalization params for {col} not found")
            
            col_min = params[col]["min"]
            col_max = params[col]["max"]
            
            if col_max - col_min > 0:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
    
    return df_normalized, params


def build_features(
    df: pd.DataFrame,
    target_col: str = "blitz",
    fit_encoders: bool = True,
    encoders: dict = None,
    normalize_params: dict = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Build feature matrix with encoding and normalization.
    
    Args:
        df: Cleaned DataFrame
        target_col: Name of target column
        fit_encoders: Whether to fit new encoders
        encoders: Existing encoders
        normalize_params: Existing normalization parameters
        
    Returns:
        Tuple of (features_df, encoders_dict, normalize_params_dict)
    """
    logger.info("Building feature matrix...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    categorical_cols = [
        "offense_personnel",
        "defense_personnel",
        "formation",
    ]
    numeric_cols = [
        "down",
        "ydstogo",
        "yardline_100",
        "quarter",
        "game_seconds_remaining",
        "score_differential",
        "shotgun",
        "motion",
    ]
    
    # Filter to columns that exist
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    
    # Encode categorical features
    X, encoders_dict = encode_categorical_features(
        X, categorical_cols, fit_encoders=fit_encoders, encoders=encoders
    )
    
    # Create derived features
    X = create_derived_features(X)
    
    # Normalize numeric features (including derived ones)
    all_numeric = numeric_cols + [
        col for col in X.columns if X[col].dtype in ["float64", "int64"]
    ]
    all_numeric = list(set(all_numeric))
    
    X, norm_params = normalize_numeric_features(
        X, all_numeric, normalize_params=normalize_params, fit=fit_encoders
    )
    
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    return X, encoders_dict, norm_params


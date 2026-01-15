"""Data loading module for NFLfastR data"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_nfl_pbp(
    seasons: list,
    columns: list,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load NFL play-by-play data from NFLfastR.
    
    Args:
        seasons: List of seasons to load (e.g., [2021, 2022, 2023])
        columns: List of columns to extract
        cache_dir: Optional directory to cache data (not used with nfl_data_py)
        
    Returns:
        DataFrame with selected columns
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py not installed. Install with: pip install nfl_data_py"
        )

    logger.info(f"Loading NFL PBP data for seasons: {seasons}")
    
    try:
        # Load play-by-play data - try loading without cache first (downloads directly)
        pbp = nfl.import_pbp_data(seasons, cache=False)
        logger.info(f"Loaded {len(pbp)} plays")
        
        # Filter to only pass/run plays (offensive plays)
        pbp_filtered = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
        logger.info(f"Filtered to {len(pbp_filtered)} offensive plays")
        
        # Engineer blitz feature: 5+ pass rushers = blitz
        if "number_of_pass_rushers" in pbp_filtered.columns:
            pbp_filtered["blitz"] = (pbp_filtered["number_of_pass_rushers"] >= 5).astype(int)
            logger.info(f"Created blitz feature from number_of_pass_rushers")
        else:
            logger.warning("number_of_pass_rushers column not found, blitz feature cannot be created")
        
        # Select only the requested columns (which may not all exist in raw data)
        # extract_blitz_features will handle missing columns
        return pbp_filtered
        
    except Exception as e:
        logger.error(f"Error loading NFL PBP data: {e}")
        raise


def extract_blitz_features(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """
    Extract blitz model features from raw PBP data.
    
    Maps old column names to nfl_data_py column names automatically.
    
    Args:
        df: Raw PBP DataFrame
        required_columns: Columns needed for blitz model
        
    Returns:
        DataFrame with required columns and clean data
    """
    logger.info("Extracting blitz features...")
    
    # Column mapping: old names -> new names from nfl_data_py
    column_mapping = {
        'quarter': 'qtr',
        'formation': 'pass_location',
        'motion': 'no_huddle',
    }
    
    # Map old column names to new ones if needed
    mapped_columns = []
    for col in required_columns:
        if col in column_mapping:
            mapped_columns.append(column_mapping[col])
        else:
            mapped_columns.append(col)
    
    # Check which columns are available
    missing_cols = [col for col in mapped_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    # Select available columns
    available_cols = [col for col in mapped_columns if col in df.columns]
    blitz_df = df[available_cols].copy()
    
    # Rename columns back to standard names for consistency
    rename_dict = {v: k for k, v in column_mapping.items()}
    # Only rename if the mapping exists and the column exists in our result
    rename_dict = {k: v for k, v in rename_dict.items() if k in blitz_df.columns}
    blitz_df = blitz_df.rename(columns=rename_dict)
    
    logger.info(f"Extracted features shape: {blitz_df.shape}")
    
    return blitz_df


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw data from CSV file"""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


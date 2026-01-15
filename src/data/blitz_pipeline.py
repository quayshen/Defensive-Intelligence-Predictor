"""Blitz model data pipeline orchestration"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.config import (
    BLITZ_COLUMNS,
    BLITZ_TARGET,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)
from src.data.load_data import load_nfl_pbp, extract_blitz_features
from src.data.clean_data import clean_blitz_data, validate_blitz_data, get_class_distribution

logger = logging.getLogger(__name__)


def data_acquisition(seasons: list = [2021, 2022, 2023]) -> Tuple[pd.DataFrame, dict]:
    """
    Execute data acquisition pipeline for blitz model.
    
    Pipeline steps:
    1. Load NFL PBP data from NFLfastR
    2. Extract blitz features (required columns only)
    3. Clean data (handle missing values, remove invalid rows)
    4. Validate data quality
    5. Save cleaned data to processed directory
    
    Args:
        seasons: List of NFL seasons to load (default: 2021-2023)
        
    Returns:
        Tuple of (cleaned_dataframe, class_distribution_dict)
        
    Example:
        >>> df, class_dist = data_acquisition([2022, 2023])
        >>> print(f"Loaded {len(df)} plays")
        >>> print(f"Class distribution: {class_dist}")
    """
    
    logger.info("=" * 70)
    logger.info("DATA ACQUISITION PIPELINE: BLITZ MODEL")
    logger.info("=" * 70)
    
    # Step 1: Load raw data
    logger.info(f"\nStep 1: Loading NFL PBP data for seasons {seasons}...")
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    pbp_raw = load_nfl_pbp(
        seasons=seasons,
        columns=BLITZ_COLUMNS,
    )
    logger.info(f"✓ Loaded {len(pbp_raw)} plays")
    
    # Step 2: Extract features
    logger.info("\nStep 2: Extracting blitz features...")
    pbp_features = extract_blitz_features(pbp_raw, BLITZ_COLUMNS)
    logger.info(f"✓ Extracted {pbp_features.shape[1]} features from {len(pbp_features)} plays")
    
    # Step 3: Clean data
    logger.info("\nStep 3: Cleaning data...")
    pbp_cleaned = clean_blitz_data(pbp_features, target_col=BLITZ_TARGET)
    logger.info(f"✓ Cleaned data shape: {pbp_cleaned.shape}")
    
    # Step 4: Validate
    logger.info("\nStep 4: Validating data...")
    validate_blitz_data(pbp_cleaned, BLITZ_COLUMNS)
    logger.info("✓ Data validation passed")
    
    # Step 5: Get class distribution
    logger.info("\nStep 5: Analyzing class distribution...")
    class_dist = get_class_distribution(pbp_cleaned, target_col=BLITZ_TARGET)
    
    # Step 6: Save
    logger.info("\nStep 6: Saving cleaned data...")
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    output_file = PROCESSED_DATA_PATH / "blitz_data_cleaned.csv"
    pbp_cleaned.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to: {output_file}")
    
    # Save metadata
    _save_dataset_metadata(pbp_cleaned, output_file)
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    return pbp_cleaned, class_dist


# Maintain backward compatibility
phase1_data_acquisition = data_acquisition


def _save_dataset_metadata(df: pd.DataFrame, data_file: Path) -> None:
    """
    Save dataset metadata for reference.
    
    Args:
        df: Cleaned DataFrame
        data_file: Path to data file
    """
    info_file = data_file.parent / "blitz_data_metadata.txt"
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("BLITZ MODEL - DATASET METADATA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total plays:        {len(df):,}\n")
        f.write(f"Number of features: {len(df.columns) - 1}\n")
        f.write(f"Data file:          {data_file.name}\n\n")
        
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 60 + "\n")
        blitz_count = (df["blitz"] == 1).sum()
        no_blitz_count = (df["blitz"] == 0).sum()
        blitz_pct = (blitz_count / len(df)) * 100
        no_blitz_pct = (no_blitz_count / len(df)) * 100
        
        f.write(f"Blitz plays:        {blitz_count:,} ({blitz_pct:.1f}%)\n")
        f.write(f"No blitz plays:     {no_blitz_count:,} ({no_blitz_pct:.1f}%)\n\n")
        
        f.write("FEATURES\n")
        f.write("-" * 60 + "\n")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            f.write(f"{i:2d}. {col:25s} | {dtype:10s} | Nulls: {nulls}\n")
        
        f.write("\nREQUIRED COLUMNS FOR BLITZ MODEL\n")
        f.write("-" * 60 + "\n")
        for col in ["down", "ydstogo", "yardline_100", "qtr",
                    "game_seconds_remaining", "score_differential",
                    "offense_personnel", "defense_personnel", "pass_location",
                    "shotgun", "no_huddle", "blitz"]:
            status = "[OK]" if col in df.columns else "[MISS]"
            f.write(f"{status} {col}\n")
    
    logger.info(f"✓ Saved metadata to: {info_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run pipeline
    df, class_dist = data_acquisition(seasons=[2021, 2022, 2023])
    print(f"\nPipeline complete!\n")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {class_dist}")

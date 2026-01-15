"""Configuration settings"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_PATH = PROJECT_ROOT / "data" / "external"

# Model paths
MODEL_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
METRICS_PATH = REPORTS_PATH / "metrics"

# Phase 1: Blitz Model - Required columns from NFLfastR
# Note: These are derived from actual nfl_data_py columns
# 'quarter' -> 'qtr', 'formation' -> 'pass_location' (proxy), 'motion' -> not in nfl_data_py
BLITZ_COLUMNS = [
    "down",  # Down (1-4)
    "ydstogo",  # Yards to go for first down
    "yardline_100",  # Distance from own endzone
    "qtr",  # Quarter (renamed from 'quarter')
    "game_seconds_remaining",  # Seconds remaining in game
    "score_differential",  # Score differential (to calculate from data)
    "offense_personnel",  # Offensive personnel (RB, TE count etc)
    "defense_personnel",  # Defensive personnel
    "pass_location",  # Pass location (proxy for formation)
    "shotgun",  # Shotgun formation indicator
    "no_huddle",  # No huddle (proxy for motion)
    "blitz",  # Target: Blitz indicator
]

# Blitz Model Target
BLITZ_TARGET = "blitz"

# Data settings
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.5
RANDOM_STATE = 42

# Model parameters (placeholder for hyperparameters)
BLITZ_MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
}

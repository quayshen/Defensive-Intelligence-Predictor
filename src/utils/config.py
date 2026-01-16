"""Configuration settings"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Phase 1: Blitz Model - Required columns from NFLfastR
# These are derived from nfl_data_py columns
BLITZ_COLUMNS = [
    "down",  # Down (1-4)
    "ydstogo",  # Yards to go for first down
    "yardline_100",  # Distance from own endzone
    "qtr",  # Quarter
    "game_seconds_remaining",  # Seconds remaining in game
    "score_differential",  # Score differential
    "offense_personnel",  # Offensive personnel
    "defense_personnel",  # Defensive personnel
    "pass_location",  # Pass location
    "shotgun",  # Shotgun formation indicator
    "no_huddle",  # No huddle indicator
    "blitz",  # Target: Blitz indicator
]

# Blitz Model Target
BLITZ_TARGET = "blitz"

# Data settings
RANDOM_STATE = 42

# Defensive Intelligence Predictor

An NFL analytics pipeline for predicting blitz packages and defensive coverage shells using play-by-play data from NFLfastR.

## Features

- **Blitz Prediction**: Predict whether a defense will blitz on a given play
- **Coverage Prediction**: Predict defensive coverage shells (Cover 0-4)
- **Feature Engineering**: Standardized preprocessing pipeline with ColumnTransformer
- **Model Training**: RandomForest and GradientBoosting implementations

## Project Structure

```
Defensive-Intelligence-Predictor/
├── data/
│   └── processed/          # Processed data and saved models
├── notebooks/
│   └── blitz_data_acquisition.ipynb    # Data loading & integration
├── src/
│   ├── data/              # Data loading & cleaning
│   ├── models/            # Model training & evaluation
│   └── utils/             # Configuration & helpers
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Run Pipeline
1. Open `notebooks/blitz_data_acquisition.ipynb`
2. Execute cells in order to:
   - Load 2021-2023 NFL PBP data
   - Generate sample coverage dataset
   - Clean and validate data
   - Create feature preprocessing pipeline

### Files Generated

After running the notebook, you'll have:
- `feature_preprocessor.pkl` - ColumnTransformer for feature preprocessing
- `feature_names.pkl` - List of feature names after transformation
- `feature_pipeline_summary.pkl` - Pipeline metadata
- `coverages_week1.csv` - Sample coverage labels

## Data Pipeline

1. **Load**: Fetch 2021-2023 NFL play-by-play data from NFLfastR
2. **Extract**: Select blitz-relevant features
3. **Clean**: Handle missing values, validate data quality
4. **Integrate**: Merge coverage labels from external sources
5. **Preprocess**: StandardScaler + OneHotEncoder via ColumnTransformer
6. **Train**: RandomForest or GradientBoosting classifiers
7. **Evaluate**: Performance metrics, confusion matrices, probabilities

## Features Used

**Numerical**: down, ydstogo, yardline_100, quarter, game_seconds_remaining, score_differential

**Categorical**: offense_personnel, defense_personnel, formation

**Binary**: shotgun, motion

## Model Performance

Coverage Shell Distribution (from sample data):
- Cover 2: 5.3%
- Cover 3: 2.9%
- Cover 0: 2.8%
- Cover 1: 2.8%
- Cover 4: 2.4%

## Dependencies

- pandas, numpy - Data manipulation
- scikit-learn - ML models and preprocessing
- nfl_data_py - NFL data access

## Next Steps

- Implement blitz prediction model training
- Train coverage prediction classifiers
- Build inference API with saved models
- Create Streamlit visualization dashboard
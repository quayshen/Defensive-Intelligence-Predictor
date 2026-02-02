# Defensive Intelligence Predictor

An interactive ML application for predicting NFL defensive blitz packages and coverage shells. This project combines data science with a user-friendly Streamlit interface to help analysts understand defensive pre-snap reads.

## 🎯 Core Features

- **Blitz Prediction Model**: Binary classification (Blitz vs. No Blitz) using Random Forest
- **Coverage Prediction Model**: Multi-class classification (Cover 0, 1, 2, 3, 4) using Random Forest
- **Interactive Dashboard**: Real-time predictions based on game situation inputs
- **Feature Visualization**: Gauge charts, probability distributions, and performance metrics

## 📊 Two-Model Architecture

| Model | Task | Output |
|-------|------|--------|
| **Blitz Model** | Binary Classification | Blitz probability + prediction |
| **Coverage Model** | Multi-class Classification | Coverage type + confidence |

Both models use the same input features for consistency.

## 🔧 Project Structure

```
Defensive-Intelligence-Predictor/
├── data/
│   └── processed/                      # Cleaned data and trained models
├── src/
│   ├── data/
│   │   ├── load_data.py               # NFL data acquisition
│   │   ├── clean_data.py              # Data cleaning
│   │   └── blitz_pipeline.py          # Full data pipeline
│   ├── models/
│   │   ├── train.py                   # Blitz model training
│   │   ├── train_coverage.py          # Coverage model training
│   │   ├── predict.py                 # Blitz inference class
│   │   └── predict_coverage.py        # Coverage inference class
│   └── utils/
│       ├── config.py                  # Configuration & paths
│       └── helpers.py                 # Utility functions
├── app/
│   ├── app.py                         # Main Streamlit app
│   ├── components.py                  # UI components
│   └── visuals.py                     # Visualization functions
├── notebooks/                         # Jupyter notebooks for exploration
├── requirements.txt                   # Python dependencies
└── README.md
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

#### Train Blitz Model
```bash
cd src/models
python train_blitz_model.py
```

#### Train Coverage Model
```bash
python train_coverage_model.py
```

Or run directly:
```python
from src.models.train import load_preprocessed_data, train_blitz_model
from src.models.train_coverage import load_coverage_data, train_coverage_model

# Train blitz model
X, y, preprocessor, feature_names, df = load_preprocessed_data()
blitz_results = train_blitz_model(X, y, preprocessor)

# Train coverage model
X, y = load_coverage_data()
coverage_results = train_coverage_model(X, y)
```

### 3. Run the Interactive App

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`

## 📱 App Usage

1. **Left Sidebar**: Configure game situation
   - Down, Yards to Go, Field Position
   - Quarter, Time Remaining, Score Differential
   - Offensive Personnel, Defense Package, Formation
   - Shotgun & Motion indicators

2. **Main Display**: View predictions
   - Blitz probability gauge
   - Coverage type prediction with confidence
   - Game situation summary

3. **Charts**: Visualizations for deeper analysis
   - Prediction confidence metrics
   - Coverage probabilities
   - Model information

## 📊 Input Features

### Game Situation
- `down` (1-4)
- `ydstogo` (1-30)
- `yardline_100` (1-100)
- `quarter` (1-4)
- `game_seconds_remaining` (0-3600)
- `score_differential` (-35 to +35)

### Personnel & Formation
- `offense_personnel` - Offensive package (11, 12, 21, etc.)
- `defense_personnel` - Defensive package (nickel, dime, base)
- `formation` - QB formation (shotgun, under center, empty)
- `shotgun` - Shotgun indicator (binary)
- `motion` - Motion indicator (binary)

## 🤖 Model Details

### Blitz Model
- **Algorithm**: Random Forest (100 trees, max_depth=15)
- **Classes**: 0 (No Blitz), 1 (Blitz - 5+ pass rushers)
- **Metrics**: Accuracy, ROC AUC, Classification Report
- **Output**: Probability + Binary Prediction

### Coverage Model
- **Algorithm**: Random Forest (100 trees, max_depth=15)
- **Classes**: Cover 0, 1, 2, 3, 4
- **Metrics**: Accuracy, Macro ROC AUC, Classification Report
- **Output**: Coverage type + Confidence score

## 📈 Dataset

- **Source**: NFLfastR play-by-play data (2021-2023)
- **Size**: ~35,000 plays
- **Target Distribution** (Blitz):
  - No Blitz: 83.7%
  - Blitz: 16.3%

## 🛠️ Key Classes

### `BlitzPredictor`
```python
from src.models.predict import BlitzPredictor

predictor = BlitzPredictor()
predictions = predictor.predict(X)  # DataFrame with blitz_probability, blitz_prediction
```

### `CoveragePredictor`
```python
from src.models.predict_coverage import CoveragePredictor

predictor = CoveragePredictor()
predictions = predictor.predict(X)  # DataFrame with coverage_type, confidence
```

## 📚 Usage Examples

### Training Models
```python
from src.models.train import load_preprocessed_data, train_blitz_model

# Load preprocessed data
X, y, preprocessor, feature_names, df = load_preprocessed_data()

# Train model
results = train_blitz_model(
    X, y, preprocessor,
    model_type="random_forest",
    test_size=0.2
)

print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"ROC AUC: {results['roc_auc']:.4f}")
```

### Making Predictions
```python
import pandas as pd
from src.models.predict import BlitzPredictor

# Create sample play
play = pd.DataFrame({
    'down': [2],
    'ydstogo': [10],
    'yardline_100': [50],
    'quarter': [2],
    'game_seconds_remaining': [1800],
    'score_differential': [0],
    'offense_personnel': ['11'],
    'defense_personnel': ['nickel'],
    'formation': ['shotgun'],
    'shotgun': [1],
    'motion': [0]
})

# Get prediction
predictor = BlitzPredictor()
pred = predictor.predict(play)
print(f"Blitz Probability: {pred['blitz_probability'].values[0]:.1%}")
```

## 🔄 Data Pipeline

1. **Acquisition**: `src/data/load_data.py` - Fetch NFL play-by-play data
2. **Cleaning**: `src/data/clean_data.py` - Handle missing values, validate
3. **Feature Engineering**: `src/data/build_features.py` - Create derived features
4. **Preprocessing**: ColumnTransformer (StandardScaler + OneHotEncoder)
5. **Modeling**: `src/models/train.py` - Train classifiers
6. **Inference**: `src/models/predict.py` - Make predictions
7. **Visualization**: `app/app.py` - Interactive dashboard

## 📋 Dependencies

- **Data**: pandas, numpy, nfl_data_py
- **ML**: scikit-learn, scipy
- **Viz**: streamlit, plotly
- **Config**: python-dotenv

See `requirements.txt` for full list with versions.

## 🚀 Deployment

### Local Testing
```bash
streamlit run app/app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push to GitHub
2. Create account at https://streamlit.io/cloud
3. Deploy directly from repository
4. Share public link

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/app.py"]
```

## 📝 Next Steps

- [ ] Add team-specific prediction comparisons
- [ ] Implement feature importance explanations
- [ ] Add confusion matrix visualization
- [ ] Create evaluation dashboard
- [ ] Add model retraining capability
- [ ] Deploy to cloud platform

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional model types (XGBoost, LightGBM)
- Advanced feature engineering
- Real-time data updates
- API endpoint development
- Mobile app interface

## 📄 License

See LICENSE file for details.

## 👤 Author

Created for portfolio demonstration of ML + full-stack skills.

---

**Status**: ✅ Interactive app complete | Models trained | Ready for deployment

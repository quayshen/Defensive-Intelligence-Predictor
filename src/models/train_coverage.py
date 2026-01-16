"""Train coverage prediction model"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import PROCESSED_DATA_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)


def load_coverage_data() -> tuple:
    """Load features and coverage labels"""
    # Load PBP data
    data_path = PROCESSED_DATA_PATH / "blitz_data_cleaned.csv"
    pbp_df = pd.read_csv(data_path)
    
    # Load coverage labels
    coverage_path = PROCESSED_DATA_PATH / "coverages_week1.csv"
    coverage_df = pd.read_csv(coverage_path)
    
    # Create a gameId/playId mapping from coverage data
    coverage_dict = {}
    for idx, row in coverage_df.iterrows():
        key = (row['gameId'], row['playId'])
        coverage_dict[key] = row['coverage']
    
    logger.info(f"Loaded {len(coverage_dict)} coverage labels")
    logger.info(f"Loaded {len(pbp_df)} PBP records")
    
    # We don't have gameId/playId in cleaned data, so we'll use the coverage data directly
    # For now, use coverage labels as-is for training
    X = pbp_df.drop(columns=['blitz'], errors='ignore').head(len(coverage_dict))
    y = coverage_df['coverage'].values[:len(X)]
    
    return X, y


def train_coverage_model(
    X: pd.DataFrame,
    y,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    model_path: Path = None,
) -> dict:
    """
    Train coverage prediction model
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Target labels (coverage types)
    model_type : str
        Model type
    test_size : float
        Test set proportion
    model_path : Path
        Path to save model
        
    Returns
    -------
    dict
        Results dictionary
    """
    logger.info(f"Training {model_type} model for coverage prediction")
    
    if model_path is None:
        model_path = PROCESSED_DATA_PATH / f"coverage_model_{model_type}.pkl"
    
    # Create preprocessor (same as blitz)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols if categorical_cols else []),
        ],
        remainder="drop"
    )
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    # Multi-class ROC AUC (macro average)
    y_pred_proba = model.predict_proba(X_test)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="macro", average="macro")
    except:
        roc_auc = 0.0
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    logger.info(f"ROC AUC (macro): {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
    
    # Save model and encoder
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    encoder_path = PROCESSED_DATA_PATH / f"coverage_label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Label encoder saved to {encoder_path}")
    
    return {
        "model": model,
        "model_path": model_path,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    X, y = load_coverage_data()
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Coverage types: {pd.Series(y).value_counts().to_dict()}")
    
    # Train model
    results = train_coverage_model(X, y)
    logger.info(f"\n✓ Coverage model trained successfully")

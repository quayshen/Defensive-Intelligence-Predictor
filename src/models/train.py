"""Train blitz prediction model"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.utils.config import PROCESSED_DATA_PATH, BLITZ_TARGET, RANDOM_STATE

logger = logging.getLogger(__name__)


def load_preprocessed_data(preprocessor_path: Path = None) -> tuple:
    """Load preprocessed features and target"""
    # Load cleaned data
    data_path = PROCESSED_DATA_PATH / "blitz_data_cleaned.csv"
    df = pd.read_csv(data_path)
    
    # Get features and target
    X = df.drop(columns=[BLITZ_TARGET], errors="ignore")
    y = df[BLITZ_TARGET] if BLITZ_TARGET in df.columns else None
    
    # Create preprocessor that handles numeric and categorical features
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
    
    # Fit preprocessor
    try:
        preprocessor.fit(X)
    except Exception as e:
        logger.warning(f"Could not fit preprocessor: {e}, using identity transform")
        from sklearn.preprocessing import FunctionTransformer
        preprocessor = FunctionTransformer()
        preprocessor.fit(X)
    
    # Load feature names if available
    try:
        with open(PROCESSED_DATA_PATH / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except:
        feature_names = list(X.columns)
    
    return X, y, preprocessor, feature_names, df


def train_blitz_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    model_path: Path = None,
) -> dict:
    """
    Train blitz prediction model
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    preprocessor
        Fitted preprocessor (ColumnTransformer)
    model_type : str
        Model type: 'random_forest', 'gradient_boosting', or 'logistic_regression'
    test_size : float
        Test set proportion
    model_path : Path
        Path to save model
        
    Returns
    -------
    dict
        Results dictionary with metrics and model
    """
    logger.info(f"Training {model_type} model for blitz prediction")
    
    if model_path is None:
        model_path = PROCESSED_DATA_PATH / f"blitz_model_{model_type}.pkl"
    
    # Transform features using preprocessor
    X_transformed = preprocessor.transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Create model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    return {
        "model": model,
        "model_path": model_path,
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "X_test": X_test,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    X, y, preprocessor, feature_names, df = load_preprocessed_data()
    
    if y is not None:
        # Train models
        results = {}
        for model_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
            results[model_type] = train_blitz_model(
                X, y, preprocessor, model_type=model_type
            )
            logger.info(f"\n{'='*60}\n")
    else:
        logger.error("Target variable not found in data")

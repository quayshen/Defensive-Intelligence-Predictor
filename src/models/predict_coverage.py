"""Make coverage predictions"""

import logging
import pickle
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


class CoveragePredictor:
    """Coverage prediction interface"""
    
    def __init__(self, model_path: Path = None, preprocessor_path: Path = None):
        """
        Initialize predictor
        
        Parameters
        ----------
        model_path : Path
            Path to trained model
        preprocessor_path : Path
            Path to feature preprocessor
        """
        if model_path is None:
            model_path = PROCESSED_DATA_PATH / "coverage_model_random_forest.pkl"
        
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None
        
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing objects"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        
        logger.info(f"Loaded model from {self.model_path}")
        
        # Create preprocessor on the fly
        try:
            data_path = PROCESSED_DATA_PATH / "blitz_data_cleaned.csv"
            df = pd.read_csv(data_path)
            
            # Use only first 13600 rows (same as training data)
            X = df.drop(columns=["blitz"], errors="ignore").iloc[:13600]
            
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            
            numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                     categorical_cols if categorical_cols else []),
                ],
                remainder="drop"
            )
            self.preprocessor.fit(X)
            logger.info("Preprocessor created and fitted")
        except Exception as e:
            logger.warning(f"Could not create preprocessor: {e}")
            raise
        
        # Load label encoder
        encoder_path = PROCESSED_DATA_PATH / "coverage_label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded")
        else:
            logger.warning("Label encoder not found")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict coverage type
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        pd.DataFrame
            Predictions with columns: 'coverage_type', 'confidence'
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
        
        # Transform features
        X_transformed = self.preprocessor.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_transformed)
        y_proba = self.model.predict_proba(X_transformed).max(axis=1)
        
        # Decode labels
        if self.label_encoder:
            coverage_types = self.label_encoder.inverse_transform(y_pred)
        else:
            coverage_types = y_pred
        
        # Return results
        return pd.DataFrame({
            "coverage_type": coverage_types,
            "confidence": y_proba,
        })
    
    def predict_single(self, X_dict: dict) -> dict:
        """
        Predict coverage for single play
        
        Parameters
        ----------
        X_dict : dict
            Single play as dictionary
            
        Returns
        -------
        dict
            Prediction result
        """
        X = pd.DataFrame([X_dict])
        result = self.predict(X)
        return result.iloc[0].to_dict()


def batch_predict(X: pd.DataFrame, model_path: Path = None) -> pd.DataFrame:
    """
    Batch prediction utility
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    model_path : Path
        Path to trained model
        
    Returns
    -------
    pd.DataFrame
        Predictions
    """
    predictor = CoveragePredictor(model_path=model_path)
    return predictor.predict(X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Make predictions
    try:
        data_path = PROCESSED_DATA_PATH / "blitz_data_cleaned.csv"
        df = pd.read_csv(data_path)
        
        X_sample = df.drop(columns=["blitz"], errors="ignore").head(10)
        
        predictor = CoveragePredictor()
        predictions = predictor.predict(X_sample)
        
        print("Sample Coverage Predictions:")
        print(predictions.head())
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Run src/models/train_coverage.py first to train the model")

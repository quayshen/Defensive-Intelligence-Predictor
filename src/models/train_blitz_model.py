"""Quick script to train and test blitz prediction"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.models.train import load_preprocessed_data, train_blitz_model
from src.models.predict import BlitzPredictor

logger = logging.getLogger(__name__)


def main():
    """Train model and test predictions"""
    
    logger.info("=" * 70)
    logger.info("BLITZ PREDICTION MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n1. Loading preprocessed data...")
    X, y, preprocessor, feature_names, df = load_preprocessed_data()
    logger.info(f"   Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    logger.info("\n2. Training Random Forest model...")
    results = train_blitz_model(X, y, preprocessor, model_type="random_forest")
    
    logger.info(f"\n   ✓ Model saved to: {results['model_path']}")
    logger.info(f"   ✓ Train accuracy: {results['train_accuracy']:.4f}")
    logger.info(f"   ✓ Test accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"   ✓ ROC AUC: {results['roc_auc']:.4f}")
    
    # Test predictions
    logger.info("\n3. Testing predictions...")
    predictor = BlitzPredictor()
    test_predictions = predictor.predict(X.head(5))
    logger.info("\n   First 5 predictions:")
    for idx, row in test_predictions.iterrows():
        logger.info(f"   Play {idx}: {row['blitz_probability']:.4f} probability, "
                   f"prediction={'BLITZ' if row['blitz_prediction'] else 'NO BLITZ'}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ MODEL TRAINED AND READY TO PREDICT BLITZ!")
    logger.info("=" * 70)
    logger.info("\nUsage:")
    logger.info("  from src.models.predict import BlitzPredictor")
    logger.info("  predictor = BlitzPredictor()")
    logger.info("  predictions = predictor.predict(X_features)")


if __name__ == "__main__":
    main()

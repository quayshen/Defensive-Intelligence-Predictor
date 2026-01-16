"""Quick script to train and test coverage prediction"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.models.train_coverage import load_coverage_data, train_coverage_model
from src.models.predict_coverage import CoveragePredictor

logger = logging.getLogger(__name__)


def main():
    """Train model and test predictions"""
    
    logger.info("=" * 70)
    logger.info("COVERAGE PREDICTION MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n1. Loading coverage data...")
    X, y = load_coverage_data()
    logger.info(f"   Features: {X.shape[1]} columns, {len(X)} samples")
    logger.info(f"   Coverage distribution:")
    import pandas as pd
    for cov_type, count in pd.Series(y).value_counts().items():
        pct = count / len(y) * 100
        logger.info(f"     - {cov_type}: {count} ({pct:.1f}%)")
    
    # Train model
    logger.info("\n2. Training Random Forest model...")
    results = train_coverage_model(X, y)
    
    logger.info(f"\n   ✓ Model saved to: {results['model_path']}")
    logger.info(f"   ✓ Train accuracy: {results['train_accuracy']:.4f}")
    logger.info(f"   ✓ Test accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"   ✓ ROC AUC: {results['roc_auc']:.4f}")
    
    # Test predictions
    logger.info("\n3. Testing predictions...")
    predictor = CoveragePredictor()
    # Use same data as was used for training
    test_predictions = predictor.predict(X.iloc[:5])
    logger.info("\n   First 5 coverage predictions:")
    for idx, row in test_predictions.iterrows():
        logger.info(f"   Play {idx}: {row['coverage_type']} ({row['confidence']:.2%} confidence)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ COVERAGE MODEL TRAINED AND READY!")
    logger.info("=" * 70)
    logger.info("\nUsage:")
    logger.info("  from src.models.predict_coverage import CoveragePredictor")
    logger.info("  predictor = CoveragePredictor()")
    logger.info("  predictions = predictor.predict(X_features)")


if __name__ == "__main__":
    main()

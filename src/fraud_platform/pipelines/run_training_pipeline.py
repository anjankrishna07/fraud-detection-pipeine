"""End-to-end training pipeline orchestration."""

import sys
from pathlib import Path

from fraud_platform.config import Config
from fraud_platform.features.build_features import build_features
from fraud_platform.ingestion.load_raw import load_raw_data
from fraud_platform.logging import get_logger
from fraud_platform.training.train import train_pipeline
from fraud_platform.validation.run_validation import validate_data

logger = get_logger(__name__)


def run_training_pipeline(
    use_kaggle_api: bool = False,
    skip_validation: bool = False,
) -> int:
    """
    Run complete training pipeline: ingest -> validate -> features -> train.

    Args:
        use_kaggle_api: Whether to download from Kaggle API
        skip_validation: Skip data validation step

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Data Ingestion
        logger.info("\n[1/4] Data Ingestion")
        logger.info("-" * 60)
        transaction_df, identity_df = load_raw_data(use_kaggle_api=use_kaggle_api)
        logger.info("✓ Data ingestion completed")

        # Step 2: Data Validation
        if not skip_validation:
            logger.info("\n[2/4] Data Validation")
            logger.info("-" * 60)
            validation_passed = validate_data()
            if not validation_passed:
                logger.error("Data validation failed. Aborting pipeline.")
                return 1
            logger.info("✓ Data validation passed")
        else:
            logger.info("\n[2/4] Data Validation (SKIPPED)")

        # Step 3: Feature Engineering
        logger.info("\n[3/4] Feature Engineering")
        logger.info("-" * 60)
        features_path = Config.DATA_FEATURES / "features.parquet"
        features_df = build_features(
            transaction_df=transaction_df,
            identity_df=identity_df,
            save_path=features_path,
        )
        logger.info("✓ Feature engineering completed")

        # Step 4: Model Training
        logger.info("\n[4/4] Model Training")
        logger.info("-" * 60)
        model, threshold = train_pipeline(
            features_path=features_path,
            experiment_name="fraud_detection",
        )
        logger.info(f"✓ Model training completed (threshold: {threshold:.4f})")

        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1


def main():
    """CLI entrypoint for training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end training pipeline")
    parser.add_argument(
        "--use-kaggle-api",
        action="store_true",
        help="Download data from Kaggle API",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation step",
    )

    args = parser.parse_args()

    return run_training_pipeline(
        use_kaggle_api=args.use_kaggle_api,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    sys.exit(main())


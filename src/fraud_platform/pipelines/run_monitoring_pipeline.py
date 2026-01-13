"""Monitoring pipeline orchestration."""

import sys
from pathlib import Path

import pandas as pd

from fraud_platform.config import Config
from fraud_platform.logging import get_logger
from fraud_platform.monitoring.drift_report import generate_drift_report
from fraud_platform.monitoring.performance_report import generate_performance_report
from fraud_platform.monitoring.retrain_trigger import check_retrain_conditions

logger = get_logger(__name__)


def run_monitoring_pipeline(
    reference_data_path: Path = None,
    current_data_path: Path = None,
) -> int:
    """
    Run monitoring pipeline: drift detection and performance reporting.

    Args:
        reference_data_path: Path to reference dataset (training data)
        current_data_path: Path to current dataset (production data)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("=" * 60)
    logger.info("Starting Monitoring Pipeline")
    logger.info("=" * 60)

    try:
        # Load reference data (training data)
        if reference_data_path is None:
            reference_data_path = Config.DATA_FEATURES / "features.parquet"

        if not reference_data_path.exists():
            logger.error(f"Reference data not found: {reference_data_path}")
            return 1

        logger.info(f"Loading reference data from {reference_data_path}")
        reference_df = pd.read_parquet(reference_data_path)

        # Load current data (production predictions/logs)
        if current_data_path is None:
            # Try to load from prediction logs
            from fraud_platform.monitoring.log_predictions import get_prediction_logger

            logger_instance = get_prediction_logger()
            recent_predictions = logger_instance.load_recent_predictions(
                n=Config.MONITORING_WINDOW_SIZE
            )

            if not recent_predictions:
                logger.warning("No recent predictions found for monitoring")
                return 1

            # Convert to DataFrame
            current_df = pd.DataFrame(recent_predictions)
        else:
            if not current_data_path.exists():
                logger.error(f"Current data not found: {current_data_path}")
                return 1
            current_df = pd.read_parquet(current_data_path)

        logger.info(f"Reference data: {len(reference_df):,} rows")
        logger.info(f"Current data: {len(current_df):,} rows")

        # Step 1: Data Drift Detection
        logger.info("\n[1/3] Data Drift Detection")
        logger.info("-" * 60)
        drift_report_path = Config.REPORTS_DIR / "drift_report.html"
        drift_report = generate_drift_report(
            reference_data=reference_df,
            current_data=current_df,
            output_path=drift_report_path,
        )
        logger.info("✓ Drift report generated")

        # Step 2: Performance Monitoring (if labels available)
        logger.info("\n[2/3] Performance Monitoring")
        logger.info("-" * 60)
        if Config.TARGET_COL in current_df.columns and "prediction" in current_df.columns:
            performance_report_path = Config.REPORTS_DIR / "performance_report.html"
            performance_report = generate_performance_report(
                reference_data=reference_df,
                current_data=current_df,
                output_path=performance_report_path,
            )
            logger.info("✓ Performance report generated")
        else:
            logger.info("⚠ Labels not available, skipping performance monitoring")

        # Step 3: Retrain Trigger Check
        logger.info("\n[3/3] Retrain Trigger Check")
        logger.info("-" * 60)
        retrain_decision = check_retrain_conditions(
            reference_data=reference_df,
            current_data=current_df,
        )

        if retrain_decision["should_retrain"]:
            logger.warning("RETRAINING RECOMMENDED")
            for reason in retrain_decision["reasons"]:
                logger.warning(f"  - {reason}")
        else:
            logger.info("No retraining needed")

        logger.info("\n" + "=" * 60)
        logger.info("Monitoring Pipeline Completed")
        logger.info("=" * 60)

        return 0 if not retrain_decision["should_retrain"] else 1

    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {e}", exc_info=True)
        return 1


def main():
    """CLI entrypoint for monitoring pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run monitoring pipeline")
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="Path to reference dataset (Parquet)",
    )
    parser.add_argument(
        "--current-data",
        type=Path,
        help="Path to current dataset (Parquet)",
    )

    args = parser.parse_args()

    return run_monitoring_pipeline(
        reference_data_path=args.reference_data,
        current_data_path=args.current_data,
    )


if __name__ == "__main__":
    sys.exit(main())


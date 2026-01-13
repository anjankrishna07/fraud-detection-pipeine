"""Retraining trigger based on drift and performance degradation."""

from pathlib import Path
from typing import Optional

import pandas as pd

from fraud_platform.config import Config
from fraud_platform.logging import get_logger
from fraud_platform.monitoring.drift_report import check_feature_drift
from fraud_platform.monitoring.performance_report import generate_performance_report

logger = get_logger(__name__)


def check_retrain_conditions(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    reference_performance: dict = None,
    current_performance: dict = None,
) -> dict:
    """
    Check if retraining conditions are met.

    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (production data)
        reference_performance: Reference performance metrics
        current_performance: Current performance metrics

    Returns:
        Dictionary with retrain decision and reasons
    """
    logger.info("Checking retraining conditions")

    retrain_decision = {
        "should_retrain": False,
        "reasons": [],
        "drift_detected": False,
        "performance_degraded": False,
    }

    # Check for data drift
    drift_summary = check_feature_drift(reference_data, current_data)
    if drift_summary["drift_detected"]:
        retrain_decision["drift_detected"] = True
        retrain_decision["reasons"].append(
            f"Data drift detected in {len(drift_summary['drifted_features'])} features"
        )

    # Check for performance degradation
    if reference_performance and current_performance:
        # Compare key metrics
        metrics_to_check = ["recall", "pr_auc", "roc_auc"]

        for metric in metrics_to_check:
            ref_value = reference_performance.get(metric)
            curr_value = current_performance.get(metric)

            if ref_value is not None and curr_value is not None:
                degradation = (ref_value - curr_value) / ref_value
                if degradation > Config.PERFORMANCE_DECAY_THRESHOLD:
                    retrain_decision["performance_degraded"] = True
                    retrain_decision["reasons"].append(
                        f"{metric} degraded by {degradation:.1%} "
                        f"({ref_value:.4f} -> {curr_value:.4f})"
                    )

    # Decision logic: retrain if drift OR significant performance degradation
    if retrain_decision["drift_detected"] or retrain_decision["performance_degraded"]:
        retrain_decision["should_retrain"] = True

    if retrain_decision["should_retrain"]:
        logger.warning(
            f"RETRAINING TRIGGERED: {', '.join(retrain_decision['reasons'])}"
        )
    else:
        logger.info("No retraining needed at this time")

    return retrain_decision


def main():
    """CLI entrypoint for retrain trigger check."""
    import argparse

    parser = argparse.ArgumentParser(description="Check retraining conditions")
    parser.add_argument(
        "--reference-data",
        type=Path,
        required=True,
        help="Path to reference dataset (Parquet)",
    )
    parser.add_argument(
        "--current-data",
        type=Path,
        required=True,
        help="Path to current dataset (Parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save retrain decision JSON",
    )

    args = parser.parse_args()

    try:
        # Load data
        reference_df = pd.read_parquet(args.reference_data)
        current_df = pd.read_parquet(args.current_data)

        # Check retrain conditions
        decision = check_retrain_conditions(reference_df, current_df)

        # Save decision
        if args.output:
            import json

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(decision, f, indent=2)
            logger.info(f"Retrain decision saved to {args.output}")

        # Exit with code 0 if retrain needed, 1 otherwise
        return 0 if decision["should_retrain"] else 1

    except Exception as e:
        logger.error(f"Retrain check failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    import sys

    sys.exit(main())


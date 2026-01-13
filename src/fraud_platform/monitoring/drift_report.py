"""Data drift detection using Evidently."""

from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = None,
) -> Report:
    """
    Generate data drift report comparing reference and current data.

    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (production data)
        output_path: Optional path to save HTML report

    Returns:
        Evidently Report object
    """
    logger.info("Generating data drift report")

    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = Config.TARGET_COL if Config.TARGET_COL in reference_data.columns else None
    column_mapping.prediction = None  # We'll add prediction separately if available

    # Create drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    # Save HTML report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drift_report.save_html(str(output_path))
        logger.info(f"Drift report saved to {output_path}")

    # Log drift summary
    try:
        drift_metrics = drift_report.as_dict()["metrics"]
        drift_detected = any(
            metric.get("result", {}).get("drift_detected", False)
            for metric in drift_metrics
        )
        logger.info(f"Data drift detected: {drift_detected}")
    except Exception as e:
        logger.warning(f"Could not parse drift metrics: {e}")

    return drift_report


def check_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = Config.DRIFT_THRESHOLD,
) -> dict:
    """
    Check for feature drift and return summary.

    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        threshold: Drift detection threshold

    Returns:
        Dictionary with drift summary
    """
    report = generate_drift_report(reference_data, current_data)

    # Parse drift results
    drift_summary = {
        "drift_detected": False,
        "drifted_features": [],
        "drift_scores": {},
    }

    try:
        metrics = report.as_dict()["metrics"]
        for metric in metrics:
            if "result" in metric:
                result = metric["result"]
                if "drift_detected" in result:
                    if result["drift_detected"]:
                        drift_summary["drift_detected"] = True
                        # Extract feature name if available
                        feature_name = metric.get("metric", "unknown")
                        drift_summary["drifted_features"].append(feature_name)
    except Exception as e:
        logger.warning(f"Error parsing drift results: {e}")

    return drift_summary


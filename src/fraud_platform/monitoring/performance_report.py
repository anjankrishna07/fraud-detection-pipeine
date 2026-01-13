"""Performance monitoring and reporting."""

from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def generate_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str = Config.TARGET_COL,
    prediction_col: str = "prediction",
    output_path: Path = None,
) -> Report:
    """
    Generate performance report comparing reference and current model performance.

    Note: This assumes labels are available (with delay in production).

    Args:
        reference_data: Reference dataset with labels
        current_data: Current dataset with labels
        target_col: Target column name
        prediction_col: Prediction column name
        output_path: Optional path to save HTML report

    Returns:
        Evidently Report object
    """
    logger.info("Generating performance report")

    # Ensure both datasets have required columns
    required_cols = [target_col, prediction_col]
    missing_ref = [col for col in required_cols if col not in reference_data.columns]
    missing_curr = [col for col in required_cols if col not in current_data.columns]

    if missing_ref:
        raise ValueError(f"Reference data missing columns: {missing_ref}")
    if missing_curr:
        raise ValueError(f"Current data missing columns: {missing_curr}")

    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col

    # Create performance report
    performance_report = Report(metrics=[ClassificationPreset()])
    performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    # Save HTML report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        performance_report.save_html(str(output_path))
        logger.info(f"Performance report saved to {output_path}")

    return performance_report


def simulate_label_delay(
    predictions_df: pd.DataFrame,
    delay_days: int = 7,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Simulate label delay in production data.

    In production, labels arrive with delay. This function filters
    predictions to only include those with available labels.

    Args:
        predictions_df: DataFrame with predictions and timestamps
        delay_days: Label delay in days
        time_col: Timestamp column name

    Returns:
        DataFrame with delayed labels
    """
    if time_col not in predictions_df.columns:
        logger.warning(f"Timestamp column {time_col} not found, skipping delay simulation")
        return predictions_df

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(predictions_df[time_col]):
        predictions_df[time_col] = pd.to_datetime(predictions_df[time_col])

    # Calculate cutoff date (delay_days ago)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=delay_days)

    # Filter to predictions with available labels
    delayed_df = predictions_df[predictions_df[time_col] <= cutoff_date].copy()

    logger.info(
        f"Simulated {delay_days}-day label delay: "
        f"{len(delayed_df)}/{len(predictions_df)} predictions have labels"
    )

    return delayed_df


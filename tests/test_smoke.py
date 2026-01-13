"""Smoke tests for end-to-end pipeline."""

import pytest
from pathlib import Path

from fraud_platform.config import Config


def test_config_loaded():
    """Test that configuration loads correctly."""
    from fraud_platform.config import Config

    assert Config.PROJECT_ROOT.exists()
    assert Config.DATA_RAW is not None
    assert Config.MLFLOW_MODEL_NAME == "fraud_detector"


def test_logging_setup():
    """Test that logging can be set up."""
    from fraud_platform.logging import get_logger

    logger = get_logger(__name__)
    assert logger is not None


def test_ingestion_module():
    """Test that ingestion module can be imported."""
    from fraud_platform.ingestion import load_raw, kaggle_download

    assert load_raw is not None
    assert kaggle_download is not None


def test_validation_module():
    """Test that validation module can be imported."""
    from fraud_platform.validation import run_validation

    assert run_validation is not None


def test_features_module():
    """Test that features module can be imported."""
    from fraud_platform.features import build_features

    assert build_features is not None


def test_training_module():
    """Test that training module can be imported."""
    from fraud_platform.training import train, thresholding

    assert train is not None
    assert thresholding is not None


def test_serving_module():
    """Test that serving module can be imported."""
    from fraud_platform.serving import app, predict, schemas

    assert app is not None
    assert predict is not None
    assert schemas is not None


def test_monitoring_module():
    """Test that monitoring module can be imported."""
    from fraud_platform.monitoring import (
        drift_report,
        log_predictions,
        performance_report,
        retrain_trigger,
    )

    assert drift_report is not None
    assert log_predictions is not None
    assert performance_report is not None
    assert retrain_trigger is not None


def test_pipelines_module():
    """Test that pipelines module can be imported."""
    from fraud_platform.pipelines import (
        run_monitoring_pipeline,
        run_training_pipeline,
    )

    assert run_training_pipeline is not None
    assert run_monitoring_pipeline is not None


@pytest.mark.skipif(
    not (Config.DATA_RAW / Config.TRANSACTION_FILE).exists(),
    reason="Raw data files not found",
)
def test_data_files_exist():
    """Test that raw data files exist (if available)."""
    transaction_path = Config.DATA_RAW / Config.TRANSACTION_FILE
    identity_path = Config.DATA_RAW / Config.IDENTITY_FILE

    # This test only runs if files exist
    if transaction_path.exists() and identity_path.exists():
        assert transaction_path.exists()
        assert identity_path.exists()


def test_api_schemas():
    """Test that API schemas can be instantiated."""
    from fraud_platform.serving.schemas import (
        BatchTransactionRequest,
        HealthResponse,
        TransactionRequest,
    )

    # Test TransactionRequest
    tx_request = TransactionRequest(
        TransactionID=123,
        TransactionAmt=100.0,
        TransactionDT=1000000,
    )
    assert tx_request.TransactionID == 123
    assert tx_request.TransactionAmt == 100.0

    # Test HealthResponse
    health = HealthResponse(status="healthy", model_loaded=False)
    assert health.status == "healthy"
    assert health.model_loaded is False


def test_threshold_optimization():
    """Test threshold optimization logic."""
    import numpy as np
    from fraud_platform.training.thresholding import find_optimal_threshold

    # Create dummy data
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_pred_proba, max_fpr=0.5)
    assert 0.0 <= threshold <= 1.0


def test_prediction_logger():
    """Test prediction logging functionality."""
    from fraud_platform.monitoring.log_predictions import PredictionLogger

    logger = PredictionLogger()
    logger.log_prediction(
        transaction_id=123,
        features={"amount": 100.0},
        fraud_probability=0.1,
        is_fraud=False,
    )

    # Load recent predictions
    predictions = logger.load_recent_predictions(n=10)
    assert isinstance(predictions, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Model evaluation utilities."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from fraud_platform.training.train import evaluate_model, prepare_features
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def evaluate_model_from_mlflow(
    model_name: str,
    stage: str = "Production",
    features_path: Path = None,
) -> dict:
    """
    Evaluate a model from MLflow registry.

    Args:
        model_name: MLflow model name
        stage: Model stage (Production, Staging, etc.)
        features_path: Path to features for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    import mlflow

    mlflow.set_tracking_uri("file:./mlruns")

    # Load model from registry
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.lightgbm.load_model(model_uri)

    logger.info(f"Loaded model {model_name} from stage {stage}")

    # Load features
    if features_path is None:
        from fraud_platform.config import Config
        features_path = Config.DATA_FEATURES / "features.parquet"

    df = pd.read_parquet(features_path)

    # Prepare features
    X, y = prepare_features(df)

    # Get threshold from model metadata or use default
    threshold = 0.5  # Default, should be stored with model

    # Evaluate
    metrics = evaluate_model(model, X, y, threshold)

    return metrics


"""Prediction logic for fraud detection."""

from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    """Fraud detection predictor that loads models from MLflow."""

    def __init__(
        self,
        model_name: str = Config.MLFLOW_MODEL_NAME,
        stage: str = "Production",
    ):
        """
        Initialize predictor.

        Args:
            model_name: MLflow model name
            stage: Model stage to load
        """
        self.model_name = model_name
        self.stage = stage
        self.model: Optional[lgb.Booster] = None
        self.threshold: float = 0.5
        self.model_version: Optional[str] = None
        self.feature_columns: Optional[list] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model from MLflow registry."""
        try:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

            # Try to load from registry
            model_uri = f"models:/{self.model_name}/{self.stage}"
            self.model = mlflow.lightgbm.load_model(model_uri)

            # Get model version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                self.model_name,
                stages=[self.stage],
            )
            if latest_version:
                self.model_version = str(latest_version[0].version)

            # Get feature columns from model
            if hasattr(self.model, "feature_name"):
                self.feature_columns = list(self.model.feature_name())
            elif hasattr(self.model, "feature_name_"):
                self.feature_columns = list(self.model.feature_name_())

            # Try to get threshold from model metadata
            try:
                model_info = client.get_model_version(
                    self.model_name,
                    self.model_version or "1",
                )
                # Threshold might be stored in tags or metadata
                # For now, use default
                self.threshold = 0.5
            except Exception:
                self.threshold = 0.5

            logger.info(
                f"Loaded model {self.model_name} v{self.model_version} "
                f"from stage {self.stage}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model from MLflow: {e}. "
                f"Predictions will not be available until model is trained."
            )
            self.model = None

    def predict(
        self,
        transaction_data: dict,
    ) -> tuple[float, bool]:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction_data: Dictionary of transaction features

        Returns:
            Tuple of (fraud_probability, is_fraud)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please train a model first.")

        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])

        # Select and align features
        if self.feature_columns:
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(df.columns)
            if missing_features:
                # Fill missing features with default values
                for feat in missing_features:
                    df[feat] = 0.0  # Default to 0 for missing features

            # Select features in the correct order
            X = df[self.feature_columns]
        else:
            # Fallback: use numeric columns
            X = df.select_dtypes(include=[np.number])

        # Predict
        fraud_probability = float(self.model.predict(X)[0])
        is_fraud = fraud_probability >= self.threshold

        return fraud_probability, is_fraud

    def predict_batch(
        self,
        transactions: list[dict],
    ) -> list[tuple[float, bool]]:
        """
        Predict fraud probability for multiple transactions.

        Args:
            transactions: List of transaction feature dictionaries

        Returns:
            List of (fraud_probability, is_fraud) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please train a model first.")

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Select and align features
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(df.columns)
            for feat in missing_features:
                df[feat] = 0.0
            X = df[self.feature_columns]
        else:
            X = df.select_dtypes(include=[np.number])

        # Predict
        fraud_probabilities = self.model.predict(X)
        predictions = [
            (float(prob), prob >= self.threshold)
            for prob in fraud_probabilities
        ]

        return predictions


# Global predictor instance
_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


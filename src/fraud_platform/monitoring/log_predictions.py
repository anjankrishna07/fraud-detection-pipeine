"""Log predictions for monitoring and analysis."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


class PredictionLogger:
    """Log predictions with timestamps for monitoring."""

    def __init__(self, log_dir: Path = None):
        """
        Initialize prediction logger.

        Args:
            log_dir: Directory to store prediction logs
        """
        if log_dir is None:
            log_dir = Config.DATA_PROCESSED / "prediction_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        transaction_id: int,
        features: Dict[str, Any],
        fraud_probability: float,
        is_fraud: bool,
        model_version: str = None,
        threshold: float = None,
    ) -> None:
        """
        Log a single prediction.

        Args:
            transaction_id: Transaction ID
            features: Transaction features
            fraud_probability: Predicted fraud probability
            is_fraud: Binary fraud prediction
            model_version: Model version used
            threshold: Decision threshold used
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": transaction_id,
            "features": features,
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "model_version": model_version,
            "threshold": threshold,
        }

        # Append to daily log file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"predictions_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def load_recent_predictions(
        self,
        n: int = 1000,
    ) -> list[Dict[str, Any]]:
        """
        Load recent predictions from log files.

        Args:
            n: Number of recent predictions to load

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        # Get all log files, sorted by date (newest first)
        log_files = sorted(self.log_dir.glob("predictions_*.jsonl"), reverse=True)

        for log_file in log_files:
            if len(predictions) >= n:
                break

            with open(log_file, "r") as f:
                for line in f:
                    if len(predictions) >= n:
                        break
                    predictions.append(json.loads(line.strip()))

        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x["timestamp"], reverse=True)

        return predictions[:n]


# Global logger instance
_prediction_logger: PredictionLogger = None


def get_prediction_logger() -> PredictionLogger:
    """Get or create global prediction logger."""
    global _prediction_logger
    if _prediction_logger is None:
        _prediction_logger = PredictionLogger()
    return _prediction_logger


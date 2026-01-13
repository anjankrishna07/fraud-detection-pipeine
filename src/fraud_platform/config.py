"""Configuration management for the fraud detection platform."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the fraud detection platform."""

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Data directories
    DATA_ROOT = PROJECT_ROOT / "data"
    DATA_RAW = Path(os.getenv("DATA_RAW_DIR", str(DATA_ROOT / "raw")))
    DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", str(DATA_ROOT / "processed")))
    DATA_FEATURES = Path(os.getenv("DATA_FEATURES_DIR", str(DATA_ROOT / "features")))

    # Kaggle credentials
    KAGGLE_USERNAME: Optional[str] = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY: Optional[str] = os.getenv("KAGGLE_KEY")

    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")
    MLFLOW_MODEL_NAME = "fraud_detector"

    # Dataset configuration
    # European Cardholders Credit Card Fraud Detection Dataset
    KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
    DATA_FILE = "creditcard.csv"  # Single CSV file for this dataset
    TRANSACTION_FILE = "creditcard.csv"  # Alias for compatibility
    IDENTITY_FILE = None  # Not applicable for this dataset

    # Training configuration
    TARGET_COL = "Class"  # 0 = non-fraud, 1 = fraud
    TIME_COL = "Time"  # Seconds elapsed since first transaction
    AMOUNT_COL = "Amount"  # Transaction amount
    VALIDATION_SPLIT = 0.2  # 20% for validation
    TEST_SPLIT = 0.1  # 10% for test
    RANDOM_STATE = 42

    # Model configuration
    MAX_FPR = 0.05  # Maximum false positive rate (5%)
    MIN_RECALL = 0.80  # Minimum recall target

    # Serving configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # Monitoring configuration
    MONITORING_WINDOW_SIZE = int(os.getenv("MONITORING_WINDOW_SIZE", "1000"))
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    PERFORMANCE_DECAY_THRESHOLD = 0.1  # 10% performance degradation triggers retrain

    # Great Expectations
    GE_CONTEXT_ROOT = PROJECT_ROOT / "great_expectations"

    # Reports directory
    REPORTS_DIR = PROJECT_ROOT / "reports"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.DATA_FEATURES,
            cls.REPORTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def has_kaggle_credentials(cls) -> bool:
        """Check if Kaggle credentials are available."""
        return cls.KAGGLE_USERNAME is not None and cls.KAGGLE_KEY is not None


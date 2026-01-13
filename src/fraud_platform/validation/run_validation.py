"""Run data validation using Great Expectations."""

import sys
from pathlib import Path

import pandas as pd

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def validate_transaction_data(df: pd.DataFrame) -> bool:
    """
    Validate transaction data against expectations.

    Args:
        df: Transaction dataframe

    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating transaction data schema and constraints")

    # Schema checks
    amount_col = Config.AMOUNT_COL if hasattr(Config, 'AMOUNT_COL') else "Amount"
    required_cols = [Config.TARGET_COL, Config.TIME_COL, amount_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Data type checks
    if not pd.api.types.is_numeric_dtype(df[Config.TIME_COL]):
        logger.error(f"{Config.TIME_COL} must be numeric")
        return False

    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        logger.error(f"{amount_col} must be numeric")
        return False

    # Range checks
    if df[amount_col].min() < 0:
        logger.warning("Found negative transaction amounts")
        # Not necessarily invalid, but worth noting

    if df[amount_col].max() > 1e6:
        logger.warning("Found very large transaction amounts (>1M)")

    # Missingness checks
    if df[Config.TIME_COL].isna().any():
        logger.error(f"{Config.TIME_COL} contains missing values")
        return False

    if df[amount_col].isna().any():
        logger.error(f"{amount_col} contains missing values")
        return False

    # Target distribution check
    if Config.TARGET_COL in df.columns:
        fraud_rate = df[Config.TARGET_COL].mean()
        logger.info(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
        if fraud_rate > 0.01:  # More than 1% is unusual
            logger.warning("Fraud rate is unusually high")

    logger.info("Transaction data validation passed")
    return True


def validate_identity_data(df: pd.DataFrame) -> bool:
    """
    Validate identity data against expectations.

    Args:
        df: Identity dataframe (may be empty for single-file datasets)

    Returns:
        True if validation passes, False otherwise
    """
    # For European Cardholders dataset, identity data is not applicable
    if df.empty or len(df.columns) == 0:
        logger.info("No identity data to validate (single-file dataset)")
        return True

    logger.info("Validating identity data schema and constraints")

    # Check for TransactionID (join key) - only if identity data exists
    if "TransactionID" not in df.columns:
        logger.warning("Identity data missing TransactionID column (may be expected for single-file datasets)")
        return True  # Not an error for single-file datasets

    # Check for duplicate TransactionIDs
    if df["TransactionID"].duplicated().any():
        logger.warning("Found duplicate TransactionIDs in identity data")

    logger.info("Identity data validation passed")
    return True


def validate_data(
    transaction_path: Path = None,
    identity_path: Path = None,
) -> bool:
    """
    Validate raw data files.

    Args:
        transaction_path: Path to transaction CSV
        identity_path: Path to identity CSV

    Returns:
        True if validation passes, False otherwise
    """
    if transaction_path is None:
        transaction_path = Config.DATA_RAW / Config.DATA_FILE
    
    if not transaction_path.exists():
        logger.error(f"Data file not found: {transaction_path}")
        return False

    # Identity file is optional (not used for European Cardholders dataset)
    identity_path_provided = identity_path is not None
    if identity_path is None and Config.IDENTITY_FILE:
        identity_path = Config.DATA_RAW / Config.IDENTITY_FILE
    
    if identity_path and not identity_path.exists() and identity_path_provided:
        logger.warning(f"Identity file not found: {identity_path} (optional for single-file datasets)")
        identity_path = None

    logger.info(f"Loading data from {transaction_path}")
    if identity_path:
        logger.info(f"Also loading identity data from {identity_path}")

    transaction_df = pd.read_csv(transaction_path, low_memory=False)
    
    if identity_path and identity_path.exists():
        identity_df = pd.read_csv(identity_path, low_memory=False)
    else:
        identity_df = pd.DataFrame()  # Empty for single-file datasets

    # Validate both datasets
    transaction_valid = validate_transaction_data(transaction_df)
    identity_valid = validate_identity_data(identity_df)

    if not transaction_valid or not identity_valid:
        logger.error("Data validation failed")
        return False

    logger.info("All data validation checks passed")
    return True


def main():
    """CLI entrypoint for data validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate raw fraud detection data")
    parser.add_argument(
        "--transaction-file",
        type=Path,
        help="Path to transaction CSV file",
    )
    parser.add_argument(
        "--identity-file",
        type=Path,
        help="Path to identity CSV file",
    )

    args = parser.parse_args()

    try:
        success = validate_data(
            transaction_path=args.transaction_file,
            identity_path=args.identity_file,
        )
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


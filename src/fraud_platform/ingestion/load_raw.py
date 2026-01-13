"""Load raw data from local files or Kaggle API."""

import sys
from pathlib import Path

import pandas as pd

from fraud_platform.config import Config
from fraud_platform.ingestion.kaggle_download import download_dataset, prompt_for_credentials
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def load_raw_data(
    use_kaggle_api: bool = False,
    username: str = None,
    key: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and identity data.

    For European Cardholders dataset: returns (transaction_df, empty_df)
    since it's a single CSV file.

    Args:
        use_kaggle_api: Whether to download from Kaggle API
        username: Kaggle username (if using API)
        key: Kaggle API key (if using API)

    Returns:
        Tuple of (transaction_df, identity_df)
        For single-file datasets, identity_df is empty
    """
    Config.ensure_directories()

    transaction_path = Config.DATA_RAW / Config.DATA_FILE

    # Download from Kaggle if requested and credentials available
    if use_kaggle_api:
        try:
            if username and key:
                download_dataset(username=username, key=key)
            elif Config.has_kaggle_credentials():
                download_dataset()
            else:
                logger.warning("Kaggle credentials not found in environment")
                username, key = prompt_for_credentials()
                if username and key:
                    download_dataset(username=username, key=key)
                else:
                    logger.info("Falling back to local files")
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            logger.info("Falling back to local files")

    # Load from local files
    if not transaction_path.exists():
        error_msg = (
            f"Required data file not found: {transaction_path}\n\n"
            "Please either:\n"
            "  1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            "  2. Place the raw CSV file in the data/raw directory\n"
            "  3. Run with --use-kaggle-api flag and provide credentials interactively"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading data from {transaction_path}")
    transaction_df = pd.read_csv(transaction_path, low_memory=False)

    # For European Cardholders dataset, there's no separate identity file
    # Create empty identity dataframe for compatibility
    identity_df = pd.DataFrame()

    logger.info(
        f"Loaded {len(transaction_df):,} transactions"
    )

    return transaction_df, identity_df


def main():
    """CLI entrypoint for data ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Load raw fraud detection data")
    parser.add_argument(
        "--use-kaggle-api",
        action="store_true",
        help="Download data from Kaggle API",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Kaggle username (if not in env vars)",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Kaggle API key (if not in env vars)",
    )

    args = parser.parse_args()

    try:
        transaction_df, identity_df = load_raw_data(
            use_kaggle_api=args.use_kaggle_api,
            username=args.username,
            key=args.key,
        )
        logger.info("Data ingestion completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


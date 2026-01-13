"""Kaggle API integration for downloading IEEE-CIS fraud dataset."""

import os
from pathlib import Path
from typing import Optional

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def setup_kaggle_credentials(username: str, key: str) -> None:
    """
    Set up Kaggle API credentials.

    Args:
        username: Kaggle username
        key: Kaggle API key
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
    kaggle_json.chmod(0o600)  # Read/write for owner only

    logger.info("Kaggle credentials configured")


def download_dataset(
    dataset: str = Config.KAGGLE_DATASET,
    output_dir: Optional[Path] = None,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> Path:
    """
    Download dataset from Kaggle using API.

    Args:
        dataset: Kaggle dataset name
        output_dir: Output directory for downloaded files
        username: Kaggle username (if not in env)
        key: Kaggle API key (if not in env)

    Returns:
        Path to downloaded data directory
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Install with: pip install kaggle"
        )

    if output_dir is None:
        output_dir = Config.DATA_RAW

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up credentials
    if username and key:
        setup_kaggle_credentials(username, key)
    elif Config.has_kaggle_credentials():
        setup_kaggle_credentials(Config.KAGGLE_USERNAME, Config.KAGGLE_KEY)
    else:
        raise ValueError(
            "Kaggle credentials not provided. "
            "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, "
            "or pass username and key as arguments."
        )

    # Download dataset
    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading dataset '{dataset}' to {output_dir}")
    
    try:
        # Try competition download first (for competition datasets)
        if dataset.startswith("c/"):
            competition_name = dataset.split("/")[-1]
            logger.info(f"Attempting to download competition: {competition_name}")
            api.competition_download_files(
                competition=competition_name,
                path=str(output_dir),
            )
            # Unzip downloaded files manually
            import zipfile
            import os
            for file in os.listdir(output_dir):
                if file.endswith('.zip'):
                    zip_path = output_dir / file
                    logger.info(f"Unzipping {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    zip_path.unlink()  # Remove zip file after extraction
        else:
            # Regular dataset download
            api.dataset_download_files(
                dataset=dataset,
                path=str(output_dir),
                unzip=True,
            )
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            competition_name = dataset.split("/")[-1] if "/" in dataset else dataset
            logger.error(
                f"403 Forbidden error. This usually means:\n"
                f"  1. You need to accept the competition terms at: "
                f"     https://www.kaggle.com/c/{competition_name}/rules\n"
                f"  2. Your API key may not have proper permissions\n"
                f"  3. The competition may require manual acceptance\n"
                f"\nPlease visit the competition page and accept the terms, then try again."
            )
        raise

    logger.info(f"Dataset downloaded successfully to {output_dir}")
    return output_dir


def prompt_for_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Prompt user for Kaggle credentials interactively.

    Returns:
        Tuple of (username, key) or (None, None) if skipped
    """
    print("\n" + "=" * 60)
    print("Kaggle API Credentials Required")
    print("=" * 60)
    print("To download the dataset automatically, you need Kaggle API credentials.")
    print("Get them from: https://www.kaggle.com/settings -> API section")
    print("\nOptions:")
    print("[1] Provide credentials now")
    print("[2] Skip API download and use local data")
    print("=" * 60)

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        username = input("Kaggle username: ").strip()
        key = input("Kaggle API key: ").strip()
        if username and key:
            return username, key
        else:
            logger.warning("Empty credentials provided, skipping download")
            return None, None
    else:
        logger.info("Skipping Kaggle API download")
        return None, None


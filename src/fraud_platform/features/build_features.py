"""Feature engineering pipeline for fraud detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_platform.config import Config
from fraud_platform.ingestion.load_raw import load_raw_data
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def frequency_encode(
    df: pd.DataFrame,
    col: str,
    prefix: str = None,
) -> pd.Series:
    """
    Frequency encode a categorical column.

    Args:
        df: Dataframe containing the column
        col: Column name to encode
        prefix: Prefix for the encoded column name

    Returns:
        Encoded series
    """
    if prefix is None:
        prefix = f"{col}_freq"

    freq_map = df[col].value_counts().to_dict()
    encoded = df[col].map(freq_map)
    encoded.name = prefix

    return encoded


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from Time column.

    Assumes Time is seconds since first transaction.

    Args:
        df: Dataframe with Time column

    Returns:
        Dataframe with temporal features added
    """
    df = df.copy()
    dt_col = Config.TIME_COL
    amount_col = Config.AMOUNT_COL if hasattr(Config, 'AMOUNT_COL') else "Amount"

    # Extract time components (assuming seconds since first transaction)
    # Convert to datetime-like features
    df[f"{dt_col}_hour"] = (df[dt_col] // 3600) % 24
    df[f"{dt_col}_day"] = (df[dt_col] // 86400) % 7  # Day of week
    df[f"{dt_col}_month"] = (df[dt_col] // (86400 * 30)) % 12  # Approximate month

    # Rolling statistics (using a window of transactions)
    df = df.sort_values(dt_col)
    df[f"{dt_col}_rolling_mean_100"] = (
        df[amount_col].rolling(window=100, min_periods=1).mean()
    )
    df[f"{dt_col}_rolling_std_100"] = (
        df[amount_col].rolling(window=100, min_periods=1).std()
    )

    logger.info("Created temporal features")
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction amount normalization features.

    Args:
        df: Dataframe with Amount column

    Returns:
        Dataframe with amount features added
    """
    df = df.copy()
    amount_col = Config.AMOUNT_COL if hasattr(Config, 'AMOUNT_COL') else "Amount"

    # Log transform (handle zeros)
    df[f"{amount_col}_log"] = np.log1p(df[amount_col])

    # Z-score normalization
    scaler = StandardScaler()
    df[f"{amount_col}_normalized"] = scaler.fit_transform(
        df[[amount_col]]
    ).flatten()

    # Binned amounts
    df[f"{amount_col}_binned"] = pd.qcut(
        df[amount_col],
        q=10,
        labels=False,
        duplicates="drop",
    )

    logger.info("Created amount features")
    return df


def create_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create missingness indicator features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with missingness features added
    """
    df = df.copy()

    # Count missing values per row
    df["missing_count"] = df.isnull().sum(axis=1)

    # Missing rate
    df["missing_rate"] = df["missing_count"] / len(df.columns)

    # Indicator for identity features (often missing)
    identity_cols = [col for col in df.columns if col.startswith("id_")]
    if identity_cols:
        df["identity_missing_count"] = df[identity_cols].isnull().sum(axis=1)
        df["has_identity"] = (df["identity_missing_count"] == 0).astype(int)

    logger.info("Created missingness features")
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Encode high-cardinality categorical features using frequency encoding.

    Args:
        df: Input dataframe
        categorical_cols: List of categorical columns to encode

    Returns:
        Dataframe with encoded features
    """
    df = df.copy()

    if categorical_cols is None:
        # Auto-detect categorical columns (object type or low cardinality numeric)
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                categorical_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:
                # Consider numeric columns with low cardinality as categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and df[col].nunique() < 100:
                    categorical_cols.append(col)

    # Filter to columns that exist
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # Frequency encode high-cardinality categoricals
    for col in categorical_cols:
        if df[col].nunique() > 10:  # High cardinality threshold
            encoded = frequency_encode(df, col)
            df[encoded.name] = encoded
            # Optionally drop original (keep for now)
            logger.debug(f"Frequency encoded {col} -> {encoded.name}")

    logger.info(f"Encoded {len(categorical_cols)} categorical features")
    return df


def build_features(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Build features from raw transaction and identity data.

    Args:
        transaction_df: Raw transaction dataframe
        identity_df: Raw identity dataframe
        save_path: Optional path to save features as Parquet

    Returns:
        Feature dataframe
    """
    logger.info("Starting feature engineering pipeline")

    # Join transaction and identity data (if identity data exists)
    if identity_df is not None and not identity_df.empty and "TransactionID" in identity_df.columns:
        logger.info("Joining transaction and identity data")
        df = transaction_df.merge(
            identity_df,
            on="TransactionID",
            how="left",
            suffixes=("", "_id"),
        )
        logger.info(f"Joined dataset shape: {df.shape}")
    else:
        logger.info("No identity data to join (single-file dataset)")
        df = transaction_df.copy()
        logger.info(f"Dataset shape: {df.shape}")

    # Create engineered features
    df = create_temporal_features(df)
    df = create_amount_features(df)
    df = create_missingness_features(df)

    # Encode categorical features
    df = encode_categorical_features(df)

    # Handle remaining missing values (fill with median for numeric, mode for categorical)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown", inplace=True)

    logger.info(f"Final feature dataframe shape: {df.shape}")
    logger.info(f"Total features: {len(df.columns)}")

    # Save features
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False, engine="pyarrow")
        logger.info(f"Features saved to {save_path}")

    return df


def main():
    """CLI entrypoint for feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(description="Build features from raw data")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for features (Parquet format)",
        default=Config.DATA_FEATURES / "features.parquet",
    )
    parser.add_argument(
        "--use-kaggle-api",
        action="store_true",
        help="Download data from Kaggle API if not found locally",
    )

    args = parser.parse_args()

    try:
        # Load raw data
        transaction_df, identity_df = load_raw_data(use_kaggle_api=args.use_kaggle_api)

        # Build features
        features_df = build_features(
            transaction_df=transaction_df,
            identity_df=identity_df,
            save_path=args.output,
        )

        logger.info("Feature engineering completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


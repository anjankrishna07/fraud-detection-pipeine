"""Train fraud detection model with LightGBM."""

import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_platform.config import Config
from fraud_platform.logging import get_logger
from fraud_platform.training.thresholding import find_optimal_threshold

logger = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    time_col: str = Config.TIME_COL,
    val_split: float = Config.VALIDATION_SPLIT,
    test_split: float = Config.TEST_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on temporal ordering.

    Args:
        df: Input dataframe
        time_col: Time column for ordering
        val_split: Fraction for validation set
        test_split: Fraction for test set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)

    # Calculate split indices
    train_end = int(n * (1 - val_split - test_split))
    val_end = int(n * (1 - test_split))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Time-based split: Train={len(train_df):,} "
        f"({len(train_df)/n:.1%}), "
        f"Val={len(val_df):,} ({len(val_df)/n:.1%}), "
        f"Test={len(test_df):,} ({len(test_df)/n:.1%})"
    )

    return train_df, val_df, test_df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = Config.TARGET_COL,
    time_col: str = Config.TIME_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        df: Input dataframe
        target_col: Target column name
        time_col: Time column to exclude

    Returns:
        Tuple of (features_df, target_series)
    """
    # Exclude non-feature columns
    exclude_cols = [
        "TransactionID",
        target_col,
        time_col,
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Select only numeric features for LightGBM
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].copy()
    y = df[target_col].copy()

    logger.info(f"Prepared {len(numeric_cols)} features for training")

    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> lgb.Booster:
    """
    Train LightGBM model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model parameters

    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": Config.RANDOM_STATE,
        }

    # Calculate class weights for imbalance
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / fraud_count if fraud_count > 0 else 1.0

    params["scale_pos_weight"] = scale_pos_weight

    logger.info(f"Training LightGBM with scale_pos_weight={scale_pos_weight:.2f}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    logger.info("Model training completed")
    return model


def evaluate_model(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X: Features
        y: True labels
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)

    metrics = {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return metrics


def train_pipeline(
    features_path: Path,
    experiment_name: str = "fraud_detection",
) -> tuple[lgb.Booster, float]:
    """
    Complete training pipeline with MLflow tracking.

    Args:
        features_path: Path to features Parquet file
        experiment_name: MLflow experiment name

    Returns:
        Tuple of (model, optimal_threshold)
    """
    logger.info("Starting training pipeline")

    # Load features
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    # Time-based split
    train_df, val_df, test_df = time_based_split(df)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Set up MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Train model
        model = train_model(X_train, y_train, X_val, y_val)

        # Get predictions for threshold optimization
        y_val_pred_proba = model.predict(X_val)

        # Find optimal threshold (maximize recall at FPR <= 5%)
        optimal_threshold = find_optimal_threshold(
            y_val,
            y_val_pred_proba,
            max_fpr=Config.MAX_FPR,
        )

        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate_model(model, X_val, y_val, optimal_threshold)
        logger.info(f"Validation metrics: {val_metrics}")

        # Evaluate on test set (single evaluation)
        test_metrics = evaluate_model(model, X_test, y_test, optimal_threshold)
        logger.info(f"Test metrics: {test_metrics}")

        # Log parameters
        mlflow.log_params({
            "max_fpr": Config.MAX_FPR,
            "optimal_threshold": optimal_threshold,
            "n_features": len(X_train.columns),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        })

        # Log metrics
        mlflow.log_metrics({
            "val_recall": val_metrics["recall"],
            "val_fpr": val_metrics["fpr"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_recall": test_metrics["recall"],
            "test_fpr": test_metrics["fpr"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
        })

        # Log model
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=Config.MLFLOW_MODEL_NAME,
        )

        # Log artifacts (precision-recall curve, confusion matrix)
        import matplotlib.pyplot as plt

        # PR curve
        precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Validation)")
        plt.grid(True)
        pr_curve_path = Path("pr_curve.png")
        plt.savefig(pr_curve_path)
        plt.close()
        mlflow.log_artifact(pr_curve_path)
        pr_curve_path.unlink()

        # Confusion matrix
        cm = confusion_matrix(y_val, (y_val_pred_proba >= optimal_threshold).astype(int))
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Validation)")
        plt.colorbar()
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = Path("confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        cm_path.unlink()

        logger.info("Training pipeline completed successfully")

    return model, optimal_threshold


def main():
    """CLI entrypoint for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--features",
        type=Path,
        help="Path to features Parquet file",
        default=Config.DATA_FEATURES / "features.parquet",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="MLflow experiment name",
        default="fraud_detection",
    )

    args = parser.parse_args()

    try:
        model, threshold = train_pipeline(
            features_path=args.features,
            experiment_name=args.experiment,
        )
        logger.info(f"Model trained with threshold: {threshold:.4f}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


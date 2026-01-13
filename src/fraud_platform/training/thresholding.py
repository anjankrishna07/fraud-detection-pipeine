"""Threshold optimization for fraud detection."""

import numpy as np
from sklearn.metrics import roc_curve

from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    max_fpr: float = Config.MAX_FPR,
    min_recall: float = Config.MIN_RECALL,
) -> float:
    """
    Find optimal threshold that maximizes recall subject to FPR constraint.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        max_fpr: Maximum allowed false positive rate
        min_recall: Minimum required recall

    Returns:
        Optimal threshold value
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Find thresholds that satisfy FPR constraint
    valid_mask = fpr <= max_fpr
    if not valid_mask.any():
        logger.warning(
            f"No threshold found with FPR <= {max_fpr}. "
            f"Using threshold with minimum FPR."
        )
        valid_mask = np.ones_like(fpr, dtype=bool)

    valid_thresholds = thresholds[valid_mask]
    valid_tpr = tpr[valid_mask]
    valid_fpr = fpr[valid_mask]

    if len(valid_thresholds) == 0:
        logger.warning("No valid thresholds found, using default 0.5")
        return 0.5

    # Among valid thresholds, find one that maximizes recall (TPR)
    # and meets minimum recall requirement
    recall_mask = valid_tpr >= min_recall if min_recall > 0 else np.ones_like(valid_tpr, dtype=bool)

    if recall_mask.any():
        # Use threshold with highest recall among those meeting constraints
        best_idx = np.argmax(valid_tpr[recall_mask])
        optimal_threshold = valid_thresholds[recall_mask][best_idx]
        optimal_recall = valid_tpr[recall_mask][best_idx]
        optimal_fpr = valid_fpr[recall_mask][best_idx]
    else:
        # If no threshold meets minimum recall, use the one with highest recall
        logger.warning(
            f"No threshold found with recall >= {min_recall}. "
            f"Using threshold with maximum recall."
        )
        best_idx = np.argmax(valid_tpr)
        optimal_threshold = valid_thresholds[best_idx]
        optimal_recall = valid_tpr[best_idx]
        optimal_fpr = valid_fpr[best_idx]

    logger.info(
        f"Optimal threshold: {optimal_threshold:.4f} "
        f"(Recall: {optimal_recall:.4f}, FPR: {optimal_fpr:.4f})"
    )

    return float(optimal_threshold)


"""Model registry utilities."""

import mlflow
from fraud_platform.config import Config
from fraud_platform.logging import get_logger

logger = get_logger(__name__)


def promote_model_to_stage(
    model_name: str,
    version: int = None,
    stage: str = "Production",
) -> None:
    """
    Promote a model version to a specific stage.

    Args:
        model_name: MLflow model name
        version: Model version (if None, uses latest)
        stage: Target stage (Production, Staging, Archived)
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()

    if version is None:
        # Get latest version
        latest_versions = client.get_latest_versions(model_name, stages=[])
        if not latest_versions:
            raise ValueError(f"No versions found for model {model_name}")
        version = latest_versions[0].version

    logger.info(f"Promoting model {model_name} version {version} to {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
    )

    logger.info(f"Model {model_name} v{version} promoted to {stage}")


def main():
    """CLI entrypoint for model promotion."""
    import argparse

    parser = argparse.ArgumentParser(description="Promote model to stage")
    parser.add_argument(
        "--model-name",
        type=str,
        default=Config.MLFLOW_MODEL_NAME,
        help="Model name in registry",
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Model version (default: latest)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Production", "Staging", "Archived"],
        help="Target stage",
    )

    args = parser.parse_args()

    try:
        promote_model_to_stage(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
        )
        return 0
    except Exception as e:
        logger.error(f"Model promotion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())


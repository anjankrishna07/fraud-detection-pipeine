"""FastAPI application for fraud detection serving."""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from fraud_platform.config import Config
from fraud_platform.logging import get_logger, setup_logging
from fraud_platform.serving.predict import get_predictor
from fraud_platform.serving.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

# Set up logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting fraud detection API server")
    try:
        predictor = get_predictor()
        if predictor.model is None:
            logger.warning(
                "Model not loaded. Train a model first or ensure MLflow model is available."
            )
    except Exception as e:
        logger.error(f"Error loading model during startup: {e}")

    yield

    # Shutdown
    logger.info("Shutting down fraud detection API server")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection API for financial transactions",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy" if predictor.model is not None else "degraded",
        model_loaded=predictor.model is not None,
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: TransactionRequest):
    """
    Predict fraud for a single transaction.

    Args:
        request: Transaction request

    Returns:
        Prediction response with fraud probability and binary decision
    """
    try:
        predictor = get_predictor()

        # Convert request to feature dict
        transaction_data = request.model_dump()
        transaction_data.update(transaction_data.pop("features", {}))

        # Predict
        fraud_probability, is_fraud = predictor.predict(transaction_data)

        return PredictionResponse(
            TransactionID=request.TransactionID,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            model_version=predictor.model_version,
            threshold=predictor.threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTransactionRequest):
    """
    Predict fraud for multiple transactions.

    Args:
        request: Batch transaction request

    Returns:
        Batch prediction response
    """
    try:
        predictor = get_predictor()

        # Convert requests to feature dicts
        transactions = []
        transaction_ids = []
        for tx in request.transactions:
            tx_data = tx.model_dump()
            tx_data.update(tx_data.pop("features", {}))
            transactions.append(tx_data)
            transaction_ids.append(tx.TransactionID)

        # Predict batch
        predictions = predictor.predict_batch(transactions)

        # Format responses
        prediction_responses = [
            PredictionResponse(
                TransactionID=tx_id,
                fraud_probability=prob,
                is_fraud=is_fraud,
                model_version=predictor.model_version,
                threshold=predictor.threshold,
            )
            for tx_id, (prob, is_fraud) in zip(transaction_ids, predictions)
        ]

        return BatchPredictionResponse(
            predictions=prediction_responses,
            model_version=predictor.model_version,
            threshold=predictor.threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


def main():
    """CLI entrypoint for serving API."""
    import uvicorn

    uvicorn.run(
        "fraud_platform.serving.app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main())


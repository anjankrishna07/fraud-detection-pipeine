"""Pydantic schemas for API requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single transaction prediction request."""

    TransactionID: int
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    TransactionDT: int = Field(..., description="Transaction datetime (seconds)")
    # Add other required fields as needed
    # For simplicity, we'll accept a flexible dict of features
    features: dict = Field(default_factory=dict, description="Additional transaction features")


class BatchTransactionRequest(BaseModel):
    """Batch transaction prediction request."""

    transactions: List[TransactionRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Single transaction prediction response."""

    TransactionID: int
    fraud_probability: float = Field(..., ge=0, le=1, description="Predicted fraud probability")
    is_fraud: bool = Field(..., description="Fraud prediction (binary)")
    model_version: Optional[str] = Field(None, description="Model version used")
    threshold: float = Field(..., description="Decision threshold used")


class BatchPredictionResponse(BaseModel):
    """Batch transaction prediction response."""

    predictions: List[PredictionResponse]
    model_version: Optional[str] = Field(None, description="Model version used")
    threshold: float = Field(..., description="Decision threshold used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: Optional[str] = None


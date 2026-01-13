# Fraud Detection API Architecture

## Overview

The Fraud Detection API is a FastAPI-based REST service that serves machine learning predictions for real-time fraud detection. It loads models from MLflow's model registry and provides low-latency inference.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                            │
│  (curl, Python, Streamlit, etc.)                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP POST /predict
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  /health          - Health check endpoint                 │  │
│  │  /predict         - Single transaction prediction         │  │
│  │  /predict_batch   - Batch predictions                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FraudPredictor Class                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Load model from MLflow registry                      │  │
│  │  2. Extract feature names from model                     │  │
│  │  3. Transform input data to match model features         │  │
│  │  4. Make prediction using LightGBM                       │  │
│  │  5. Apply threshold to get binary decision               │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Model: fraud_detector                                    │  │
│  │  Stage: Production                                        │  │
│  │  Version: 1                                               │  │
│  │  Type: LightGBM Booster                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LightGBM Model                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Input: 40 engineered features                            │  │
│  │  Output: Fraud probability (0.0 to 1.0)                  │  │
│  │  Threshold: 0.5 (configurable)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

### 1. API Startup (Lifespan Events)

When the API server starts:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model from MLflow
    predictor = get_predictor()  # Creates FraudPredictor instance
    # ... model loading happens here ...
    
    yield  # Server runs here
    
    # Shutdown: Cleanup
```

**What happens:**
- `FraudPredictor` is instantiated
- Model is loaded from MLflow registry (`models:/fraud_detector/Production`)
- Feature names are extracted from the model
- Model version and threshold are retrieved

### 2. Health Check Request

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1"
}
```

**Code Flow:**
1. Check if predictor has loaded model
2. Return status based on model availability
3. Include model version if available

### 3. Prediction Request

```
POST /predict
Content-Type: application/json

{
  "TransactionID": 123,
  "TransactionAmt": 149.62,
  "TransactionDT": 0,
  "features": {
    "V1": -1.36,
    "V2": -0.07,
    ...
    // All 40 features
  }
}
```

**Step-by-Step Processing:**

#### Step 1: Request Validation (Pydantic)
```python
class TransactionRequest(BaseModel):
    TransactionID: int
    TransactionAmt: float
    TransactionDT: int
    features: dict
```
- Validates request structure
- Ensures data types are correct
- Raises 422 error if validation fails

#### Step 2: Feature Extraction
```python
transaction_data = request.model_dump()
transaction_data.update(transaction_data.pop("features", {}))
```
- Merges top-level fields with features dict
- Creates a single dictionary with all transaction data

#### Step 3: Feature Alignment
```python
# Convert to DataFrame
df = pd.DataFrame([transaction_data])

# Get model's expected features
feature_columns = model.feature_name()  # Returns list of 40 features

# Fill missing features with 0.0
missing_features = set(feature_columns) - set(df.columns)
for feat in missing_features:
    df[feat] = 0.0

# Select features in correct order
X = df[feature_columns]
```
- Converts dict to pandas DataFrame
- Aligns features with model's expected order
- Fills missing features with default values (0.0)

#### Step 4: Model Prediction
```python
# LightGBM prediction
fraud_probability = model.predict(X)[0]  # Returns probability [0.0, 1.0]
```
- LightGBM model processes the 40 features
- Returns fraud probability (float between 0.0 and 1.0)

#### Step 5: Binary Decision
```python
threshold = 0.5  # From model metadata or default
is_fraud = fraud_probability >= threshold
```
- Compares probability to threshold
- Returns boolean decision

#### Step 6: Response Formation
```python
return PredictionResponse(
    TransactionID=request.TransactionID,
    fraud_probability=fraud_probability,
    is_fraud=is_fraud,
    model_version=predictor.model_version,
    threshold=predictor.threshold,
)
```

**Response:**
```json
{
  "TransactionID": 123,
  "fraud_probability": 0.00215,
  "is_fraud": false,
  "model_version": "1",
  "threshold": 0.5
}
```

## Key Components

### 1. FraudPredictor Class

**Location:** `src/fraud_platform/serving/predict.py`

**Responsibilities:**
- Model loading from MLflow
- Feature name extraction
- Prediction logic
- Threshold management

**Key Methods:**
```python
def _load_model(self):
    """Loads model from MLflow registry"""
    model_uri = f"models:/{self.model_name}/{self.stage}"
    self.model = mlflow.lightgbm.load_model(model_uri)
    self.feature_columns = list(self.model.feature_name())

def predict(self, transaction_data: dict) -> tuple[float, bool]:
    """Returns (fraud_probability, is_fraud)"""
    # Feature alignment and prediction logic
```

### 2. FastAPI Application

**Location:** `src/fraud_platform/serving/app.py`

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Error Handling:**
- 422: Validation errors (invalid request format)
- 500: Internal server errors (prediction failures)
- 503: Service unavailable (model not loaded)

### 3. Pydantic Schemas

**Location:** `src/fraud_platform/serving/schemas.py`

**Purpose:**
- Request validation
- Response serialization
- Type safety

## Model Loading Process

1. **MLflow Connection:**
   ```python
   mlflow.set_tracking_uri("file:./mlruns")
   ```

2. **Model URI Construction:**
   ```python
   model_uri = "models:/fraud_detector/Production"
   ```
   - Format: `models:/<model_name>/<stage>`
   - Stages: Production, Staging, Archived

3. **Model Loading:**
   ```python
   model = mlflow.lightgbm.load_model(model_uri)
   ```
   - Loads LightGBM Booster object
   - Includes all model artifacts

4. **Feature Extraction:**
   ```python
   features = list(model.feature_name())
   ```
   - Gets list of 40 feature names in correct order
   - Used for feature alignment during prediction

## Feature Engineering in API

**Important:** The API expects **engineered features**, not raw transaction data.

**Required Features (40 total):**
- V1-V28: PCA features from original dataset
- Amount, Time: Original transaction fields
- Time_hour, Time_day, Time_month: Temporal features
- Time_rolling_mean_100, Time_rolling_std_100: Rolling statistics
- Amount_log, Amount_normalized, Amount_binned: Amount transformations
- missing_count, missing_rate: Missingness indicators
- Time_hour_freq: Frequency encoding

**For Production:**
- Apply the same feature engineering pipeline used during training
- Store preprocessing transformers (scalers, encoders) with the model
- Use MLflow's preprocessing pipeline support

## Performance Characteristics

- **Latency:** ~10-50ms per prediction (depending on hardware)
- **Throughput:** Can handle hundreds of requests per second
- **Model Size:** ~1-5 MB (LightGBM models are compact)
- **Memory:** ~100-200 MB (model + dependencies)

## Error Scenarios

### 1. Model Not Loaded
```json
{
  "detail": "Model not loaded. Please train a model first."
}
```
**Cause:** No model in Production stage
**Solution:** Train model and promote to Production

### 2. Missing Features
**Behavior:** Missing features are filled with 0.0
**Impact:** May affect prediction accuracy
**Solution:** Ensure all 40 features are provided

### 3. Invalid Request Format
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "TransactionID"],
      "msg": "Field required"
    }
  ]
}
```
**Cause:** Request doesn't match schema
**Solution:** Check request format matches TransactionRequest schema

## Monitoring & Logging

- **Structured Logging:** All operations logged with context
- **Prediction Logging:** Can be extended to log all predictions
- **Health Monitoring:** `/health` endpoint for monitoring systems
- **Model Versioning:** Response includes model version for tracking

## Security Considerations

- **Input Validation:** Pydantic schemas validate all inputs
- **Type Safety:** Strong typing prevents type-related errors
- **Error Handling:** Errors don't expose internal details
- **Rate Limiting:** Can be added with middleware (not currently implemented)

## Future Enhancements

1. **Feature Engineering Pipeline:** Integrate preprocessing into API
2. **Model A/B Testing:** Support multiple model versions
3. **Caching:** Cache predictions for identical transactions
4. **Async Processing:** Use async/await for better concurrency
5. **Batch Processing:** Optimize batch endpoint for large requests
6. **Explainability:** Add SHAP/LIME explanations to responses



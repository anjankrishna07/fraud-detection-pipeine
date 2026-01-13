# Fraud Detection ML Platform

A complete, end-to-end machine learning platform for real-time fraud detection in financial transactions.

## Problem Statement

A financial institution processes a continuous stream of card transactions and must decide, in near real time, whether each transaction is fraudulent. Fraudulent transactions represent less than 0.5% of all events, but missing them results in direct financial loss, while incorrectly flagging legitimate transactions leads to customer dissatisfaction and operational cost.

The task is to design, implement, and deploy an end-to-end machine learning system that predicts the probability of fraud for each transaction under the following constraints:

- The system must operate on highly imbalanced data with delayed and potentially noisy labels.
- Model decisions must be explainable to support manual review by fraud analysts.
- Predictions must be served via a low-latency inference API suitable for real-time use.
- The system must monitor data and concept drift in production and retrain automatically when performance degrades.
- Model performance must be evaluated using business-aligned metrics, prioritizing high recall at a fixed false-positive rate rather than overall accuracy.

Success is defined by the system's ability to reduce expected financial loss over time while maintaining operational stability, demonstrated through offline evaluation, simulated production inference, and continuous monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Ingestion Layer                        │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │ Kaggle API   │    OR    │ Local Files │                    │
│  └──────┬───────┘         └──────┬───────┘                    │
│         │                        │                             │
│         └────────────┬───────────┘                             │
│                      ▼                                           │
│              ┌───────────────┐                                  │
│              │ Raw Data      │                                  │
│              │ (CSV)         │                                  │
│              └───────┬───────┘                                  │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Validation Layer                         │
│              ┌──────────────────────┐                           │
│              │ Great Expectations   │                           │
│              │ - Schema validation  │                           │
│              │ - Range checks       │                           │
│              │ - Missingness checks │                           │
│              └───────┬──────────────┘                           │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Engineering Layer                     │
│              ┌──────────────────────┐                           │
│              │ Feature Pipeline     │                           │
│              │ - Temporal features  │                           │
│              │ - Amount features    │                           │
│              │ - Missingness        │                           │
│              │ - Frequency encoding │                           │
│              └───────┬──────────────┘                           │
│                      │                                          │
│              ┌───────▼──────────────┐                           │
│              │ Features (Parquet)    │                          │
│              └───────┬──────────────┘                           │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Layer                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Time-based Split (Train/Val/Test)                        │   │
│  └───────────────┬──────────────────────────────────────────┘   │
│                  │                                              │
│          ┌───────▼──────────────┐                               │
│          │ LightGBM Training    │                               │
│          │ - Class weights      │                               │
│          │ - Early stopping     │                               │
│          └───────┬──────────────┘                               │
│                  │                                              │
│          ┌───────▼──────────────┐                               │
│          │ Threshold Optimization│                              │
│          │ (Recall @ FPR ≤ 5%)   │                              │
│          └───────┬──────────────┘                               │
│                  │                                              │
│          ┌───────▼──────────────┐                               │
│          │ MLflow Registry       │                              │
│          │ - Model versioning    │                              │
│          │ - Experiment tracking │                              │
│          └───────┬──────────────┘                               │
└──────────────────┼──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                               │
│              ┌──────────────────────┐                           │
│              │ FastAPI Application   │                           │
│              │ - /predict            │                           │
│              │ - /predict_batch      │                           │
│              │ - /health             │                           │
│              └───────┬──────────────┘                           │
│                      │                                           │
│              ┌───────▼──────────────┐                           │
│              │ Prediction Logging    │                           │
│              └──────────────────────┘                           │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                              │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Drift Detection  │  │ Performance      │                    │
│  │ (Evidently)      │  │ Monitoring       │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │                     │                                │
│           └──────────┬──────────┘                                │
│                     ▼                                            │
│           ┌──────────────────────┐                               │
│           │ Retrain Trigger      │                               │
│           │ - Drift threshold    │                               │
│           │ - Performance decay  │                               │
│           └──────────────────────┘                               │
└──────────────────────────────────────────────────────────────────┘
```

## Dataset

This platform uses the **European Cardholders Credit Card Fraud Detection Dataset** as a proxy for real financial transaction data. The dataset is publicly available on Kaggle and simulates the characteristics of real-world fraud detection scenarios:

- **Highly imbalanced**: Fraudulent transactions represent ~0.17% of all events (492 fraud cases out of 284,807 transactions)
- **Temporal structure**: Transactions are ordered by `Time` (seconds elapsed since the first transaction)
- **PCA features**: Contains 28 anonymized numerical features (V1-V28) obtained via Principal Component Analysis
- **Single file**: All data is contained in one CSV file (no separate identity table)

### Dataset Assumptions

- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset
- **Temporal ordering**: Used for time-based train/validation/test splits (no random splitting)
- **Amount**: Transaction amount (may need normalization)
- **Class**: Target variable (0 = non-fraudulent, 1 = fraudulent)

## Metrics and Thresholding Strategy

### Business-Aligned Metrics

The system optimizes for business outcomes rather than overall accuracy:

1. **Recall at Fixed FPR**: Maximize recall (true positive rate) while constraining false positive rate ≤ 5%
2. **PR-AUC**: Precision-Recall Area Under Curve (better for imbalanced data than ROC-AUC)
3. **ROC-AUC**: Overall model discrimination ability

### Threshold Selection

The optimal decision threshold is selected on the validation set to:
- Maximize recall (catch as many fraud cases as possible)
- Subject to constraint: FPR ≤ 5% (minimize false alarms)

This threshold is then fixed and evaluated once on the test set.

## Monitoring and Retraining

### Data Drift Detection

- **Tool**: Evidently AI
- **Method**: Statistical tests comparing reference (training) vs. current (production) data distributions
- **Frequency**: Configurable (default: per monitoring pipeline run)
- **Output**: HTML reports in `reports/drift_report.html`

### Performance Monitoring

- **Delayed Labels**: Simulates real-world scenario where fraud labels arrive with delay (default: 7 days)
- **Metrics Tracked**: Recall, Precision, FPR, PR-AUC, ROC-AUC
- **Output**: HTML reports in `reports/performance_report.html`

### Retraining Trigger

Automatic retraining is triggered when:

1. **Data Drift Detected**: Significant distribution shift in input features
2. **Performance Degradation**: >10% drop in key metrics (recall, PR-AUC, ROC-AUC)

The retrain trigger emits a clear signal and can be integrated with orchestration systems (e.g., Airflow) for automated retraining.

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**:
   ```bash
   cd /path/to/fintech
   ```

2. **Install dependencies**:
   ```bash
   make setup
   # OR
   pip install -e .
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials (see Credentials section)
   ```

### Credentials

#### Required (for automatic dataset download)

- **KAGGLE_USERNAME**: Your Kaggle username
- **KAGGLE_KEY**: Your Kaggle API key

Get credentials from: https://www.kaggle.com/settings -> API section

#### Optional

- **MLFLOW_TRACKING_URI**: MLflow tracking URI (default: `file:./mlruns`)
- **MLFLOW_ARTIFACT_ROOT**: MLflow artifact root (default: `./mlruns`)
- **API_HOST**: API server host (default: `0.0.0.0`)
- **API_PORT**: API server port (default: `8000`)

### Running Without Kaggle API

If you don't have Kaggle credentials, you can manually download the dataset and place the files in `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

The system will automatically detect and use local files if they exist.

## Usage

### Command Line Interface

The platform provides a Makefile with convenient commands:

```bash
# Install dependencies
make setup

# Download/load raw data
make ingest

# Validate raw data
make validate

# Build features
make features

# Train model
make train

# Start API server
make serve

# Run monitoring pipeline
make monitor

# Run tests
make test

# Clean generated files
make clean
```

### Python Module Interface

You can also run components directly:

```bash
# Data ingestion
python -m fraud_platform.ingestion.load_raw --use-kaggle-api

# Data validation
python -m fraud_platform.validation.run_validation

# Feature engineering
python -m fraud_platform.features.build_features

# Training
python -m fraud_platform.training.train

# Training pipeline (end-to-end)
python -m fraud_platform.pipelines.run_training_pipeline

# Monitoring pipeline
python -m fraud_platform.pipelines.run_monitoring_pipeline

# Model promotion
python -m fraud_platform.training.register_model --stage Production
```

### Docker Deployment

Build and run with Docker Compose:

```bash
# Build images
make docker-build
# OR
docker-compose build

# Run training
make docker-train
# OR
docker-compose run --rm training

# Start API server
make docker-serve
# OR
docker-compose up api

# Run monitoring
make docker-monitor
# OR
docker-compose run --rm monitoring
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 123,
    "TransactionAmt": 100.0,
    "TransactionDT": 1000000,
    "features": {}
  }'
```

Response:
```json
{
  "TransactionID": 123,
  "fraud_probability": 0.0234,
  "is_fraud": false,
  "model_version": "1",
  "threshold": 0.1234
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "TransactionID": 123,
        "TransactionAmt": 100.0,
        "TransactionDT": 1000000,
        "features": {}
      },
      {
        "TransactionID": 124,
        "TransactionAmt": 500.0,
        "TransactionDT": 1000001,
        "features": {}
      }
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {
      "TransactionID": 123,
      "fraud_probability": 0.0234,
      "is_fraud": false,
      "model_version": "1",
      "threshold": 0.1234
    },
    {
      "TransactionID": 124,
      "fraud_probability": 0.5678,
      "is_fraud": true,
      "model_version": "1",
      "threshold": 0.1234
    }
  ],
  "model_version": "1",
  "threshold": 0.1234
}
```

## Project Structure

```
fraud-ml-platform/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.training
├── src/
│   └── fraud_platform/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── kaggle_download.py
│       │   └── load_raw.py
│       ├── validation/
│       │   ├── __init__.py
│       │   └── run_validation.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── thresholding.py
│       │   └── register_model.py
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── schemas.py
│       │   └── predict.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── log_predictions.py
│       │   ├── drift_report.py
│       │   ├── performance_report.py
│       │   └── retrain_trigger.py
│       └── pipelines/
│           ├── __init__.py
│           ├── run_training_pipeline.py
│           └── run_monitoring_pipeline.py
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

## Common Issues and Troubleshooting

### Issue: Kaggle API credentials not found

**Solution**: 
1. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` file, OR
2. Place raw CSV files manually in `data/raw/` directory

### Issue: Model not found when starting API

**Solution**: Train a model first using `make train` or `python -m fraud_platform.training.train`

### Issue: Out of memory during training

**Solution**: 
- Reduce dataset size for testing
- Use data sampling in feature engineering
- Increase system memory or use cloud resources

### Issue: Great Expectations validation fails

**Solution**: 
- Check that data files are not corrupted
- Verify schema matches expected format
- Review validation logs for specific failures

### Issue: Docker build fails

**Solution**: 
- Ensure Docker is running
- Check that `pyproject.toml` is valid
- Verify Python 3.11 is available in base image

## Development

### Running Tests

```bash
make test
# OR
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

## License

MIT License

## Acknowledgments

- IEEE-CIS Fraud Detection Dataset (Kaggle)
- LightGBM for gradient boosting
- MLflow for experiment tracking
- Evidently AI for drift detection
- FastAPI for API serving


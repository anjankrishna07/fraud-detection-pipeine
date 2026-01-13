"""Example script for making prediction requests to the fraud detection API."""

import requests
import json
import pandas as pd
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/predict"

def predict_from_features_file(transaction_id: int = 1):
    """
    Make a prediction using features from the feature engineering output.
    
    Args:
        transaction_id: Index of transaction to predict (0-based)
    """
    # Load features
    features_path = Path("data/features/features.parquet")
    if not features_path.exists():
        print(f"Features file not found: {features_path}")
        return None
    
    df = pd.read_parquet(features_path)
    
    if transaction_id >= len(df):
        print(f"Transaction ID {transaction_id} out of range (max: {len(df)-1})")
        return None
    
    # Get transaction features (exclude target)
    sample = df.iloc[transaction_id]
    features = {
        col: float(sample[col]) 
        for col in df.columns 
        if col != 'Class' and pd.api.types.is_numeric_dtype(df[col])
    }
    
    # Prepare request
    request_data = {
        "TransactionID": transaction_id + 1,
        "TransactionAmt": float(sample.get("Amount", 100.0)),
        "TransactionDT": int(sample.get("Time", 0)),
        "features": features
    }
    
    # Make prediction
    response = requests.post(API_URL, json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"Transaction ID: {result['TransactionID']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Model Version: {result['model_version']}")
        print(f"Threshold: {result['threshold']:.4f}")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def predict_from_raw_data(
    transaction_id: int,
    v1: float, v2: float, v3: float, v4: float, v5: float,
    v6: float, v7: float, v8: float, v9: float, v10: float,
    v11: float, v12: float, v13: float, v14: float, v15: float,
    v16: float, v17: float, v18: float, v19: float, v20: float,
    v21: float, v22: float, v23: float, v24: float, v25: float,
    v26: float, v27: float, v28: float,
    amount: float,
    time: float = 0.0,
):
    """
    Make a prediction from raw transaction data.
    
    Note: This requires providing engineered features. For production use,
    you should apply the same feature engineering pipeline used during training.
    """
    # Basic feature engineering (simplified - should match training pipeline)
    import numpy as np
    
    # Temporal features
    time_hour = (time // 3600) % 24
    time_day = (time // 86400) % 7
    time_month = (time // (86400 * 30)) % 12
    
    # Amount features
    amount_log = np.log1p(amount)
    # Note: Amount_normalized would need the scaler from training
    # For now, using a simple approximation
    amount_normalized = (amount - 88.35) / 250.0  # Approximate mean/std from dataset
    amount_binned = min(9, max(0, int(amount / 100)))  # Simple binning
    
    # Missingness (assuming no missing values for this example)
    missing_count = 0.0
    missing_rate = 0.0
    
    # Frequency encoding (simplified)
    time_hour_freq = 1.0  # Would need actual frequency map
    
    # Rolling statistics (simplified - would need actual rolling window)
    time_rolling_mean_100 = amount  # Approximation
    time_rolling_std_100 = 0.0  # Approximation
    
    request_data = {
        "TransactionID": transaction_id,
        "TransactionAmt": amount,
        "TransactionDT": int(time),
        "features": {
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
            "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
            "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
            "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
            "V26": v26, "V27": v27, "V28": v28,
            "Amount": amount,
            "Time": time,
            "Time_hour": time_hour,
            "Time_day": time_day,
            "Time_month": time_month,
            "Time_rolling_mean_100": time_rolling_mean_100,
            "Time_rolling_std_100": time_rolling_std_100,
            "Amount_log": amount_log,
            "Amount_normalized": amount_normalized,
            "Amount_binned": amount_binned,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "Time_hour_freq": time_hour_freq,
        }
    }
    
    response = requests.post(API_URL, json=request_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Fraud Detection API - Prediction Examples")
    print("=" * 60)
    
    # Example 1: Predict from features file
    print("\n[Example 1] Predicting from features file (transaction 0):")
    print("-" * 60)
    predict_from_features_file(transaction_id=0)
    
    # Example 2: Predict from raw data
    print("\n[Example 2] Predicting from raw transaction data:")
    print("-" * 60)
    result = predict_from_raw_data(
        transaction_id=999,
        v1=-1.36, v2=-0.07, v3=2.54, v4=1.38, v5=-0.34,
        v6=0.46, v7=0.24, v8=0.10, v9=0.36, v10=0.09,
        v11=-0.55, v12=-0.62, v13=-0.99, v14=-0.31, v15=1.47,
        v16=-0.47, v17=0.21, v18=0.03, v19=0.40, v20=0.25,
        v21=-0.02, v22=0.28, v23=-0.11, v24=0.07, v25=0.13,
        v26=-0.19, v27=0.13, v28=-0.02,
        amount=149.62,
        time=0.0,
    )
    
    if result:
        print(f"✅ Prediction successful!")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Is Fraud: {result['is_fraud']}")



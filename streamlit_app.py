"""Streamlit web interface for fraud detection predictions."""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .fraud-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-fraud {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(transaction_data):
    """Make prediction request to API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def load_sample_transaction():
    """Load a sample transaction from the dataset."""
    features_path = Path("data/features/features.parquet")
    if features_path.exists():
        df = pd.read_parquet(features_path)
        sample = df.iloc[0]
        return {
            "V1": float(sample.get("V1", 0)),
            "V2": float(sample.get("V2", 0)),
            "V3": float(sample.get("V3", 0)),
            "V4": float(sample.get("V4", 0)),
            "V5": float(sample.get("V5", 0)),
            "V6": float(sample.get("V6", 0)),
            "V7": float(sample.get("V7", 0)),
            "V8": float(sample.get("V8", 0)),
            "V9": float(sample.get("V9", 0)),
            "V10": float(sample.get("V10", 0)),
            "V11": float(sample.get("V11", 0)),
            "V12": float(sample.get("V12", 0)),
            "V13": float(sample.get("V13", 0)),
            "V14": float(sample.get("V14", 0)),
            "V15": float(sample.get("V15", 0)),
            "V16": float(sample.get("V16", 0)),
            "V17": float(sample.get("V17", 0)),
            "V18": float(sample.get("V18", 0)),
            "V19": float(sample.get("V19", 0)),
            "V20": float(sample.get("V20", 0)),
            "V21": float(sample.get("V21", 0)),
            "V22": float(sample.get("V22", 0)),
            "V23": float(sample.get("V23", 0)),
            "V24": float(sample.get("V24", 0)),
            "V25": float(sample.get("V25", 0)),
            "V26": float(sample.get("V26", 0)),
            "V27": float(sample.get("V27", 0)),
            "V28": float(sample.get("V28", 0)),
            "Amount": float(sample.get("Amount", 0)),
            "Time": float(sample.get("Time", 0)),
        }
    return {}

# Main header
st.markdown('<h1 class="main-header">🔒 Fraud Detection Platform</h1>', unsafe_allow_html=True)

# Check API health
with st.sidebar:
    st.header("🔍 System Status")
    health = check_api_health()
    
    if health:
        st.success("✅ API Connected")
        st.info(f"Model Version: {health.get('model_version', 'N/A')}")
        st.info(f"Status: {health.get('status', 'N/A')}")
    else:
        st.error("❌ API Not Available")
        st.warning("Please ensure the API server is running:\n`make serve` or `python -m fraud_platform.serving.app`")
        st.stop()

# Main content
tab1, tab2 = st.tabs(["📊 Single Prediction", "📁 Batch Prediction"])

with tab1:
    st.header("Single Transaction Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        transaction_id = st.number_input("Transaction ID", min_value=1, value=1, step=1)
        amount = st.number_input("Amount", min_value=0.0, value=149.62, step=0.01, format="%.2f")
        time = st.number_input("Time (seconds)", min_value=0, value=0, step=1)
        
        # Load sample button
        if st.button("📥 Load Sample Transaction"):
            sample = load_sample_transaction()
            if sample:
                st.session_state.sample_features = sample
                st.success("Sample transaction loaded!")
                st.json(sample)
    
    with col2:
        st.subheader("PCA Features (V1-V28)")
        
        # Create feature inputs
        if 'sample_features' in st.session_state:
            features = st.session_state.sample_features.copy()
        else:
            features = {}
        
        # V1-V14 in first column
        v_cols = st.columns(2)
        with v_cols[0]:
            for i in range(1, 15):
                key = f"V{i}"
                features[key] = st.number_input(
                    f"V{i}",
                    value=float(features.get(key, 0.0)),
                    step=0.01,
                    format="%.4f",
                    key=f"v{i}"
                )
        
        with v_cols[1]:
            for i in range(15, 29):
                key = f"V{i}"
                features[key] = st.number_input(
                    f"V{i}",
                    value=float(features.get(key, 0.0)),
                    step=0.01,
                    format="%.4f",
                    key=f"v{i}"
                )
    
    # Engineered features (simplified - in production these should be computed)
    st.subheader("Engineered Features")
    eng_col1, eng_col2, eng_col3 = st.columns(3)
    
    with eng_col1:
        time_hour = st.number_input("Time Hour", min_value=0, max_value=23, value=0, step=1)
        time_day = st.number_input("Time Day", min_value=0, max_value=6, value=0, step=1)
        time_month = st.number_input("Time Month", min_value=0, max_value=11, value=0, step=1)
    
    with eng_col2:
        time_rolling_mean = st.number_input("Time Rolling Mean (100)", value=float(amount), step=0.01, format="%.2f")
        time_rolling_std = st.number_input("Time Rolling Std (100)", value=0.0, step=0.01, format="%.2f")
        amount_log = st.number_input("Amount Log", value=float(np.log1p(amount)), step=0.01, format="%.4f")
    
    with eng_col3:
        amount_normalized = st.number_input("Amount Normalized", value=0.25, step=0.01, format="%.4f")
        amount_binned = st.number_input("Amount Binned", min_value=0, max_value=9, value=8, step=1)
        missing_count = st.number_input("Missing Count", min_value=0.0, value=0.0, step=1.0)
        missing_rate = st.number_input("Missing Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
        time_hour_freq = st.number_input("Time Hour Freq", min_value=0.0, value=1.0, step=1.0)
    
    # Predict button
    if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            # Prepare request
            request_data = {
                "TransactionID": int(transaction_id),
                "TransactionAmt": float(amount),
                "TransactionDT": int(time),
                "features": {
                    **{f"V{i}": features.get(f"V{i}", 0.0) for i in range(1, 29)},
                    "Amount": float(amount),
                    "Time": float(time),
                    "Time_hour": float(time_hour),
                    "Time_day": float(time_day),
                    "Time_month": float(time_month),
                    "Time_rolling_mean_100": float(time_rolling_mean),
                    "Time_rolling_std_100": float(time_rolling_std),
                    "Amount_log": float(amount_log),
                    "Amount_normalized": float(amount_normalized),
                    "Amount_binned": float(amount_binned),
                    "missing_count": float(missing_count),
                    "missing_rate": float(missing_rate),
                    "Time_hour_freq": float(time_hour_freq),
                }
            }
            
            # Make prediction
            result = make_prediction(request_data)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display results
                fraud_prob = result.get("fraud_probability", 0.0)
                is_fraud = result.get("is_fraud", False)
                threshold = result.get("threshold", 0.5)
                
                # Result box
                box_class = "fraud-detected" if is_fraud else "no-fraud"
                st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED")
                else:
                    st.success(f"✅ No Fraud Detected")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Probability", f"{fraud_prob:.4f}", f"{(fraud_prob*100):.2f}%")
                with col2:
                    st.metric("Decision Threshold", f"{threshold:.4f}")
                with col3:
                    st.metric("Model Version", result.get("model_version", "N/A"))
                
                # Probability bar
                st.progress(fraud_prob)
                st.caption(f"Fraud probability: {fraud_prob:.4f} (Threshold: {threshold:.4f})")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show full response
                with st.expander("📋 View Full Response"):
                    st.json(result)

with tab2:
    st.header("Batch Prediction")
    st.info("Upload a CSV file with transaction data to get batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("🔮 Predict Batch", type="primary"):
                st.warning("Batch prediction endpoint coming soon. For now, use the single prediction tab.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Fraud Detection ML Platform | Model Version: {version}
    </div>
    """.format(version=health.get("model_version", "N/A") if health else "N/A"),
    unsafe_allow_html=True
)



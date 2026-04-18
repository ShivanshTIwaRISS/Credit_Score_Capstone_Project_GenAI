# Streamlit Artifact Documentation

This document explains the contents of the `dt_model_streamlit.pkl` artifact and provides exact details on how to load and use it seamlessly inside your Streamlit UI.

---

## 1. Artifact Contents

The exported `.pkl` file contains a cleanly formatted Python dictionary with four main sections.

| Section | Key | Description |
|---------|-----|-------------|
| **1. Inference Artifacts** | `model` | The fitted `DecisionTreeClassifier` model ready for `predict_proba()`. |
| | `scaler` | The fitted `StandardScaler` used to normalize the data. |
| **2. Config & Preprocessing** | `config["cat_cols"]` | List of nominal columns (e.g., `["person_home_ownership", ...]`) to one-hot encode. |
| | `config["feature_columns"]` | The ordered list of 16 post-OHE columns required by the model. |
| | `config["dt_threshold"]` | The optimal decision threshold scalar (e.g. `0.35`). |
| | `config["alias_map"]` | Dictionary mapping friendly Streamlit variable names (e.g. `"income"`) to their raw model counterparts (e.g. `"person_income($)"`). |
| | `config["defaults"]` | Training-set median/modes to securely fill missing attributes. |
| **3. Fallback Models (Optional)**| `lr_model` | Logistic Regression Model for UI comparisons. |
| | `config_lr["lr_threshold"]` | Default `0.50` threshold parameter. |
| **4. UI Metadata** | `metadata["dataset_info"]` | Number of samples/features used for training. Used for dashboard charts. |
| | `metadata["dt_metrics"]` | Test-time metrics (Accuracy, ROC-AUC) for dashboard cards. |
| | `metadata["lr_metrics"]` | Test-time metrics for the Logistic Regression Fallback. |

---

## 2. Expected Input Schema

When sending data into the preprocessing pipeline layer, your exact input schema must follow standard Python dictionary formatting mapping *either* friendly alias keys *or* actual column keys. 

**Valid Streamlit Form Inputs:**

```python
mock_input = {
    "age": 30,                             # Extracted straight from Streamlit Slider
    "income": 50000,                       # Uses friendly alias maps beautifully 
    "home_ownership": "OWN",               # Dropdown string
    "employment_years": 5.2,               # Float parameter
    "loan_amount": 10000,
    "interest_rate": 11.0, 
    "loan_intent": "PERSONAL",             # Unaliased internal identifier acts normally
    "default_on_file": "N",                
    "credit_history": 5 
}
```

> [!TIP] 
> Because the `alias_map` translates and `defaults` imputes missing strings automatically, your Streamlit app can safely drop irrelevant form factors without breaking the ML inference cycle.

---

## 3. Usage inside Streamlit (`app.py`)

Here is a copy-pasteable example of exactly how your `app.py` script should load the artifact and process an end-to-end inference request natively. 

```python
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- 1. Load Artifact ---
@st.cache_resource
def load_artifact():
    path = "dt_model_streamlit.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    st.error("Missing dt_model_streamlit.pkl")
    st.stop()

pkg = load_artifact()

# De-structure Core Dependencies
model = pkg["model"]
scaler = pkg["scaler"]
config = pkg["config"]

# --- 2. Streamlit UI Definitions ---
st.title("Credit Risk UI Prediction")
age = st.slider("Borrower Age", 18, 100, 30)
income = st.number_input("Annual Income", min_value=1)
# ... build standard Streamlit elements ...

# Assemble payload
raw_input = {
    "age": age,
    "income": income,
    "home_ownership": "RENT",
    "loan_amount": 1200,
    # ...
}

# --- 3. Run Inference Flow ---
if st.button("Predict"):
    # 1. Expand Aliases dynamically
    resolved = {config["alias_map"].get(k, k): v for k, v in raw_input.items()}
    
    # 2. Impute Defaults (Handles trailing missing variables gracefully)
    row = {col: resolved.get(col, config["defaults"][col]) for col in config["defaults"]}
    
    # Auto-computations (i.e., Loan To Income Ratio check)
    if "loan_percent_income" not in resolved:
        row["loan_percent_income"] = round(row["loan_amnt($)"] / max(row["person_income($)"], 1), 4)

    # 3. Encoding and Alignment 
    df = pd.DataFrame([row])
    df_enc = pd.get_dummies(df, columns=config["cat_cols"], drop_first=True)
    df_aligned = df_enc.reindex(columns=config["feature_columns"], fill_value=0)

    # 4. Standard Scaler Transform
    X_scaled = scaler.transform(df_aligned.values)

    # 5. Pipeline Extracted Execution
    proba = model.predict_proba(X_scaled)[0][1]
    prediction = 1 if proba >= config["dt_threshold"] else 0

    if prediction == 1:
        st.error(f"HIGH RISK (Default Prob: {proba*100:.1f}%)")
    else:
        st.success(f"APPROVED (Default Prob: {proba*100:.1f}%)")
```

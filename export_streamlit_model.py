#!/usr/bin/env python3
import pickle
import os
import pandas as pd

# Constants exact from notebook/credit_risk_per_pipeline_v5 (1).ipynb
_ALIAS_MAP = {
    "income":           "person_income($)",
    "age":              "person_age",
    "employment_years": "person_emp_length",
    "home_ownership":   "person_home_ownership",
    "loan_amount":      "loan_amnt($)",
    "interest_rate":    "loan_int_rate",
    "default_on_file":  "cb_person_default_on_file",
    "credit_history":   "cb_person_cred_hist_length",
}

_DEFAULTS = {
    "person_age":                  27,
    "person_income($)":            48000,
    "person_home_ownership":       "RENT",
    "person_emp_length":           4.0,
    "loan_intent":                 "PERSONAL",
    "loan_amnt($)":                8000,
    "loan_int_rate":               11.0,
    "loan_percent_income":         0.17,
    "cb_person_default_on_file":   "N",
    "cb_person_cred_hist_length":  3,
}

def build_streamlit_artifact():
    input_path = "dt_model.pkl"
    output_path = "dt_model_streamlit.pkl"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "rb") as f:
        pkg = pickle.load(f)

    # Re-package clearly for Streamlit
    streamlit_pkg = {
        # 1. Inference Artifacts
        "model": pkg.get("model"),
        "scaler": pkg.get("scaler"),
        
        # 2. Config & Preprocessing Logic
        "config": {
            "cat_cols": pkg.get("cat_cols", ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]),
            "feature_columns": pkg.get("feature_columns"),
            "dt_threshold": pkg.get("dt_threshold", 0.35),
            "alias_map": _ALIAS_MAP,
            "defaults": _DEFAULTS,
        },
        
        # 3. Secondary/Fallback Models (Optional)
        "lr_model": pkg.get("lr_model"),
        "config_lr": {
            "lr_threshold": pkg.get("lr_threshold", 0.50),
        },
        
        # 4. UI Metadata
        "metadata": {
            "dataset_info": pkg.get("dataset_info", {}),
            "dt_metrics": pkg.get("dt_metrics", {}),
            "lr_metrics": pkg.get("lr_metrics", {}),
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(streamlit_pkg, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Successfully exported Streamlit artifact to {output_path}")
    
    # Simple Verification
    print("Running verification...")
    # Mock Streamlit Input App
    mock_input = {"income": 50000, "age": 30, "loan_amount": 10000, "home_ownership": "OWN"}
    resolved = {streamlit_pkg["config"]["alias_map"].get(k, k): v for k, v in mock_input.items()}
    row = {col: resolved.get(col, streamlit_pkg["config"]["defaults"][col]) for col in streamlit_pkg["config"]["defaults"]}
    df = pd.DataFrame([row])
    df_enc = pd.get_dummies(df, columns=streamlit_pkg["config"]["cat_cols"], drop_first=True)
    df_aligned = df_enc.reindex(columns=streamlit_pkg["config"]["feature_columns"], fill_value=0)
    X_scaled = streamlit_pkg["scaler"].transform(df_aligned.values)
    proba = streamlit_pkg["model"].predict_proba(X_scaled)[0][1]
    
    threshold = streamlit_pkg["config"]["dt_threshold"]
    decision = "REJECT" if proba >= threshold else "APPROVE"
    print(f"Verification Successful: Proba={proba:.4f}, Decision={decision}")

if __name__ == "__main__":
    build_streamlit_artifact()

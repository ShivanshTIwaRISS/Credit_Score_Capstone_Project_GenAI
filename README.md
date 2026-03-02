# Project 4: CreditIQ — Intelligent Credit Risk Scoring System
## From Borrower Profiling to Real-Time Default Prediction

### Project Overview

This project involves the design and implementation of an **AI-powered credit risk intelligence platform** that evaluates borrower profiles and predicts loan default probabilities using classical machine learning.

- **Milestone 1:** Data cleaning, exploratory analysis, and classical ML model training applied to historical loan application data to predict credit risk and identify key default drivers.
- **Milestone 2:** Production-grade Streamlit application with multi-model support, risk banding, explainability features, and a premium dark-themed dashboard UI.

---

### Constraints & Requirements

- **Team Size:** 3–4 Students
- **Dataset:** Kaggle Credit Risk Dataset (32,500+ rows)
- **Framework:** Scikit-learn (Classical ML — no LLMs required)
- **Hosting:** Mandatory (Streamlit Cloud or equivalent)

---

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **ML Models** | Decision Tree Classifier, Logistic Regression, Scikit-Learn |
| **Data Processing** | Pandas, NumPy |
| **UI Framework** | Streamlit (Custom CSS, Dark Theme) |
| **Visualizations** | Matplotlib, Seaborn, Plotly, Altair |
| **Serialization** | Pickle |

---

### Project Structure

```text
Credit_Score_Capstone_Project_GenAI/
├── app.py                                   # Main Streamlit web application
├── dt_model.pkl                             # Serialized model pipeline (pickle)
├── requirements.txt                         # Python dependency list
├── README.md                                # This file
├── cleaned.md                               # Detailed file and dataset documentation
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset_raw.csv      # Original Kaggle dataset (32.5k rows)
│   └── cleaned/
│       └── cleaned_credit_risk.csv          # Fully cleaned & processed dataset
└── notebook/
    ├── data_cleaning.ipynb                  # Data cleaning & EDA notebook
    └── model_training.ipynb                 # ML model training & evaluation notebook
```

---

### Milestones & Deliverables

#### Milestone 1: Data Pipeline & ML Model Training (Mid-Sem)

**Objective:** Build a robust data cleaning pipeline and train interpretable ML models on historical loan data — focused on classical ML techniques *without LLMs*.

**Key Deliverables:**
- Problem understanding & business context.
- System architecture diagram.
- EDA and data cleaning notebook (`data_cleaning.ipynb`).
- Trained model pipeline serialized to `dt_model.pkl`.
- Model performance evaluation report (Accuracy, F1, ROC-AUC).

#### Milestone 2: Interactive Risk Scoring Application (End-Sem)

**Objective:** Deploy a fully interactive application that accepts applicant data in real time, scores credit risk, explains predictions, and presents model performance dashboards.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Real-time risk prediction with probability scores and risk banding (LOW / MEDIUM / HIGH).
- Simulated Loan Grade assignment (A–G) based on composite risk score.
- Multi-model comparison (Decision Tree vs. Logistic Regression).
- Feature importance explanations for each prediction.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### How It Works

**Step 1 — Data Cleaning (`data_cleaning.ipynb`)**
Raw Kaggle data is preprocessed using IQR-based clipping for outliers (e.g., impossible ages, 60+ year employment histories) and median imputation for missing interest rates and employment lengths. A simulated `loan_grade` feature is engineered for portfolio-level analysis.

**Step 2 — Model Training (`model_training.ipynb`)**
Two models are trained to balance linear insights against non-linear pattern recognition. The pipeline packages StandardScaler, LabelEncoders, evaluation metrics, and both models into a single `dt_model.pkl` artifact for deployment.

**Step 3 — Application (`app.py`)**
The Streamlit app loads the serialized pipeline, encodes categorical inputs, scales numerics, and computes default probability via `.predict_proba()`. Results are rendered as risk tiers, probability gauges, and feature importance breakdowns.

---

### Input Features

The model relies on 10 core features:

| Feature | Description |
| :--- | :--- |
| `person_age` | Applicant's age (18–100) |
| `person_income` | Annual income ($) |
| `person_home_ownership` | RENT, OWN, MORTGAGE, or OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOME IMPROVEMENT, DEBT CONSOLIDATION |
| `loan_amnt` | Loan amount requested ($) |
| `loan_int_rate` | Interest rate (%) |
| `cb_person_default_on_file` | Historical default on file (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |
| `loan_percent_income` | Auto-calculated: Loan Amount ÷ Income |

---

### Evaluation & Performance

| Model | Training Accuracy | Testing Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | ~92.9% | ~91.0% | Primary model, tuned with `max_depth=10` |
| **Logistic Regression** | ~85% | ~82.33% | Secondary benchmark model |

Comprehensive metric reports (Precision, Recall, F1) and interactive Confusion Matrices are available in the **Performance** tab of the Streamlit application.

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering, UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Prediction accuracy, Explainability quality, UI/UX polish, Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. The application must be publicly hosted.

---

### Setup & Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd Credit_Score_Capstone_Project_GenAI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.

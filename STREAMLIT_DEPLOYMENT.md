# FraudScope: Streamlit deployment guide

This folder contains a complete interactive Streamlit application for the *Financial Fraud Detection* project in this repository. It reuses the **same feature engineering** and **same XGBoost model weights** used by the original FastAPI service. It does **not** require retraining or downloading PaySim to make predictions.

## What you get

- **Overview:** Transaction counts, flagged cases, probability distribution, decisions by transaction type, highest-risk transactions
- **Check Transaction:** Interactive single-transaction form with example inputs, model scores, recommended demo decision, XGBoost local Tree SHAP explanation, JSON result download
- **Batch Screening:** Upload CSV (20 MB / up to 50,000 rows), validate each row, score valid rows, highlight invalid rows, filter actions, download enriched CSV
- **Model Insights:** Native XGBoost global split-gain feature importance, full feature list, training limitations and inconsistent source-reported metrics
- **Risk policy controls:** OTP/review/block thresholds controlled in the sidebar (these alter decisions, not the model probability)

## Repository structure

```text
Financial-Fraud-Detection/
├── app.py                                 # Streamlit entry point
├── requirements.txt                       # Streamlit-only dependencies
├── .streamlit/config.toml                 # App theme and CSV upload limit
├── fraud_ui/service.py                    # Validation, scoring and XGBoost explanations
├── examples/sample_transactions.csv       # Example CSV, including invalid rows
├── models/
│   ├── feature_columns.json
│   ├── xgboost_fraud_model.pkl             # Original FastAPI model (preserved)
│   ├── xgboost_fraud_model.json            # Original JSON (preserved)
│   └── xgboost_fraud_model_streamlit.json  # Canonical model re-export for Streamlit
├── src/features.py                        # Original feature pipeline (reused)
├── api/risk_engine.py                      # Original decision policy (reused)
└── tests/test_streamlit_service.py         # Inference and validation tests
```

## Local launch (Windows PowerShell)

**Use Python 3.12** for reproducible installs.

```powershell
# From the project folder containing app.py
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser. No API server needs to be running.

**Other operating systems:** activate the environment with `source .venv/bin/activate` and run the same install/start commands.

## Deploy from GitHub to Streamlit Community Cloud

1. Create a GitHub repository or use your existing Financial Fraud repository. Add the new `app.py`, `fraud_ui/`, `.streamlit/`, `requirements.txt`, the model artifacts (including `xgboost_fraud_model_streamlit.json`), `examples/`, tests, and this guide. Keep the existing `src/` and `api/risk_engine.py` files. You can upload the entire provided package; it excludes `.git` internals and large datasets.
2. Do **not** upload customer transaction files, private data, secrets or the large PaySim dataset.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, then choose **Create app** → **Yup, I have an app**.
4. Select repository, branch `main`, and app entrypoint **`app.py`**. Choose **Python 3.12** in Advanced settings (where available).
5. Deploy; the app is served on a Streamlit-owned `*.streamlit.app` subdomain that you can customize. If using a private repository, grant the corresponding GitHub permissions and configure the app's viewer sharing appropriately.
6. If you see `ModuleNotFoundError` or model startup errors, inspect Community Cloud logs and verify root `requirements.txt` and included model files.

Streamlit documentation:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## CSV input columns

Mandatory (case-sensitive):

```csv
type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
TRANSFER,1000,5000,4000,10000,11000
CASH_OUT,2500,2500,0,0,0
```

Optional extra columns, e.g. `transaction_id`, `step`, or labeled `isFraud`, are retained in downloads. `isFraud` is **never used as a model input**; if supplied, the Batch Screening screen can compute **optional** evaluation metrics on valid labeled rows. Only `TRANSFER` and `CASH_OUT` are supported. Uploaded rows of other types or rows with invalid fields are marked `invalid`, not labeled safe.

Outputs: `fraud_probability`, `recommended_action`, `is_fraud`, `threshold_used`, `record_status`, `validation_error`, `source_row`.

Default *example* decision thresholds: `0.30` for OTP, `0.60` for review/flagging, `0.85` for block. Change them in the sidebar; they are **not validated thresholds for real transactions**.

## Why a new model JSON file is included

The two **original** artifacts in the uploaded project, `xgboost_fraud_model.pkl` and `xgboost_fraud_model.json`, do not give identical predictions for the same transaction. The existing FastAPI service loads the `.pkl` artifact, so **that** is the reference.

The new `xgboost_fraud_model_streamlit.json` was exported by loading the canonical `.pkl` in XGBoost 3.1.3 and calling `model.save_model(...)`. Predictions from the re-export were compared against the `.pkl` on 200 generated rows and matched **exactly** (`max_abs_diff = 0`). The new app loads only the vetted Streamlit export and checks all 13 model feature names and their order.

## Safety, privacy, and limitations

- This is an **educational ML prototype** trained using **synthetic PaySim data**, not a certified payment-fraud product.
- `newbalanceOrig` and `newbalanceDest` are post-transaction balances in the original dataset. If these are unavailable before settlement, the model cannot perform claimed **pre-authorization** detection without rethinking the features. Do not use this interface to approve or block real payments.
- The original README and model card report conflicting evaluation numbers, and the held-out labeled data were not included in the uploaded ZIP. These cannot be independently confirmed from the project files alone.
- The app does not write transaction results to a database. User data is still sent to the Streamlit server for inference and held transiently in that session's server memory. **Never upload actual customer financial records to a publicly hosted demo.**
- If production deployment is desired, use a secured FastAPI backend, authentication/authorization, audit logs, encryption, proper data-retention policy, rate limits, drift monitoring, and calibrated thresholds.

## Optional tests

```powershell
pip install pytest joblib
python -m pytest -q tests/test_streamlit_service.py
```

The original FastAPI backend remains in the repository and can still be deployed independently.

### Existing GitHub Actions workflow

The **full project** retains its earlier `.github/workflows/ci-cd.yml`, which attempts AWS ECS/ECR deployment on pushes to `main`. That AWS job may fail if its secrets and infrastructure have not been configured. Streamlit Community Cloud deployment is independent of that workflow. For the simplest Streamlit-only GitHub repository, use the **standalone Streamlit archive**, which contains no AWS deployment workflow.

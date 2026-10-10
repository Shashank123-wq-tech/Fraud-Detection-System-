"""Inference and input validation for the Financial Fraud Streamlit app.

The saved model is derived from models/xgboost_fraud_model.pkl (the artifact
used by the original FastAPI service). No dataset or API server is needed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from api.risk_engine import RiskThresholds, is_fraud_decision, recommend_action
from src.features import SELECTED_FEATURES, engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgboost_fraud_model_streamlit.json"
FEATURE_PATH = ROOT / "models" / "feature_columns.json"
TRANSACTION_TYPES = ("TRANSFER", "CASH_OUT")
REQUIRED = ("type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
NUMERIC = REQUIRED[1:]
MAX_BATCH_ROWS = 50_000


@lru_cache(maxsize=1)
def load_fraud_model() -> XGBClassifier:
    """Cache a vetted, JSON-based export of the model used by FastAPI."""
    with FEATURE_PATH.open(encoding="utf-8") as f:
        feature_columns = json.load(f)
    if feature_columns != SELECTED_FEATURES:
        raise RuntimeError("Feature list mismatch; refusing to score transactions.")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    if model.n_features_in_ != len(feature_columns):
        raise RuntimeError("Model feature count does not match the deployment schema.")
    actual_names = model.get_booster().feature_names
    if actual_names is not None and actual_names != feature_columns:
        raise RuntimeError("Model feature names or order do not match training.")
    return model


def validate_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate each row without silently changing invalid financial records.

    Returns normalized original records and a row-indexed Series of error text.
    No transaction is scored unless all six model inputs are valid.
    """
    missing = [key for key in REQUIRED if key not in data.columns]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    if len(data) > MAX_BATCH_ROWS:
        raise ValueError(f"Batch limit is {MAX_BATCH_ROWS:,} rows. Split the file into smaller batches.")

    cleaned = data.copy().reset_index(drop=True)
    cleaned["type"] = cleaned["type"].astype("string").str.strip().str.upper()
    for name in NUMERIC:
        cleaned[name] = pd.to_numeric(cleaned[name], errors="coerce")

    issues: list[list[str]] = [[] for _ in range(len(cleaned))]
    invalid_type = (~cleaned["type"].isin(TRANSACTION_TYPES)).fillna(True).to_numpy(dtype=bool)
    for idx in np.flatnonzero(invalid_type):
        issues[int(idx)].append("type must be TRANSFER or CASH_OUT")

    for name in NUMERIC:
        col = cleaned[name].to_numpy(dtype=float, na_value=np.nan)
        bad = ~np.isfinite(col)
        for idx in np.flatnonzero(bad):
            issues[int(idx)].append(f"{name} must be a finite number")
        wrong_sign = np.isfinite(col) & (col <= 0 if name == "amount" else col < 0)
        for idx in np.flatnonzero(wrong_sign):
            issues[int(idx)].append(f"{name} must be {'greater than 0' if name == 'amount' else 'non-negative'}")

    errors = pd.Series(["; ".join(items) for items in issues], name="validation_error", dtype="string")
    return cleaned, errors


def probability_for_valid_rows(valid_rows: pd.DataFrame) -> np.ndarray:
    if valid_rows.empty:
        return np.empty(0, dtype=float)
    x = engineer_features(valid_rows)
    if list(x.columns) != SELECTED_FEATURES:
        raise RuntimeError("Generated features differ from training; inference aborted.")
    result = load_fraud_model().predict_proba(x)[:, 1]
    if not np.isfinite(result).all():
        raise RuntimeError("Model produced a non-finite score; inference aborted.")
    return result.astype(float)


def screen_batch(data: pd.DataFrame, thresholds: RiskThresholds | None = None) -> pd.DataFrame:
    """Keep invalid rows in the output as errors, never as safe predictions."""
    normalized, errors = validate_transactions(data)
    output = data.copy().reset_index(drop=True)
    output.insert(0, "source_row", np.arange(1, len(output) + 1))
    output["validation_error"] = errors
    output["record_status"] = np.where(errors.eq(""), "scored", "invalid")
    output["fraud_probability"] = np.nan

    valid_mask = (errors == "").to_numpy(dtype=bool)
    if valid_mask.any():
        output.loc[valid_mask, "fraud_probability"] = probability_for_valid_rows(normalized.loc[valid_mask])
    return apply_risk_policy(output, thresholds or RiskThresholds())


def apply_risk_policy(scored: pd.DataFrame, thresholds: RiskThresholds) -> pd.DataFrame:
    """Recompute actions without changing model probabilities."""
    result = scored.copy()
    if "fraud_probability" not in result.columns:
        raise ValueError("Expected fraud_probability column in scored data")
    valid = result["fraud_probability"].notna()
    prob = result.loc[valid, "fraud_probability"].astype(float)
    result["recommended_action"] = "invalid"
    result["is_fraud"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if valid.any():
        result.loc[valid, "recommended_action"] = [recommend_action(float(p), thresholds) for p in prob]
        result.loc[valid, "is_fraud"] = pd.array([is_fraud_decision(float(p), thresholds) for p in prob], dtype="boolean")
    result["threshold_used"] = thresholds.review
    return result


def explain_transaction(transaction: dict) -> pd.DataFrame:
    """Native XGBoost local Tree SHAP values in log-odds units, not causation."""
    frame = pd.DataFrame([transaction])
    normalized, errors = validate_transactions(frame)
    if errors.iloc[0]:
        raise ValueError(str(errors.iloc[0]))
    features = engineer_features(normalized)
    booster = load_fraud_model().get_booster()
    values = booster.predict(DMatrix(features, feature_names=SELECTED_FEATURES), pred_contribs=True)[0]
    return pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "feature_value": features.iloc[0].values,
        "log_odds_contribution": values[:-1],
    }).sort_values("log_odds_contribution", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def global_importance() -> pd.DataFrame:
    booster = load_fraud_model().get_booster()
    raw = booster.get_score(importance_type="gain")
    return pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "mean_split_gain": [float(raw.get(f, 0)) for f in SELECTED_FEATURES],
    }).sort_values("mean_split_gain", ascending=False).reset_index(drop=True)

"""Tests for the Streamlit app's shared scoring service (no UI runtime needed)."""
import numpy as np
import pandas as pd
import pytest

from api.risk_engine import RiskThresholds
from fraud_ui.service import (
    apply_risk_policy,
    explain_transaction,
    global_importance,
    load_fraud_model,
    screen_batch,
    validate_transactions,
)


def routine():
    return {
        "type": "TRANSFER", "amount": 1000.0,
        "oldbalanceOrg": 5000.0, "newbalanceOrig": 4000.0,
        "oldbalanceDest": 10000.0, "newbalanceDest": 11000.0,
    }


def test_model_feature_names_unchanged():
    model = load_fraud_model()
    assert model.n_features_in_ == 13
    assert len(model.get_booster().feature_names) == 13


def test_known_routine_and_account_drain():
    normal = routine()
    drained = {"type": "CASH_OUT", "amount": 1000, "oldbalanceOrg": 1000,
               "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0}
    result = screen_batch(pd.DataFrame([normal, drained]))
    assert result["record_status"].tolist() == ["scored", "scored"]
    assert result["recommended_action"].tolist() == ["approve", "block"]
    assert float(result["fraud_probability"].iloc[0]) < .3
    assert float(result["fraud_probability"].iloc[1]) > .85


def test_invalid_rows_preserved_but_not_scored():
    bad_type = dict(routine(), type="PAYMENT")
    bad_amount = dict(routine(), amount=-1)
    missing_value = dict(routine(), oldbalanceDest=np.nan)
    results = screen_batch(pd.DataFrame([routine(), bad_type, bad_amount, missing_value]))
    assert results["record_status"].tolist() == ["scored", "invalid", "invalid", "invalid"]
    assert results["fraud_probability"].isna().sum() == 3
    assert all(results.loc[1:, "recommended_action"] == "invalid")
    assert all(results.loc[1:, "validation_error"].str.len() > 0)


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="Missing required CSV columns"):
        validate_transactions(pd.DataFrame([routine()]).drop(columns=["newbalanceDest"]))


def test_new_threshold_reclassifies_without_changing_score():
    old = screen_batch(pd.DataFrame([routine()]))
    new = apply_risk_policy(old, RiskThresholds(otp=.05, review=.1, block=.2))
    assert new["fraud_probability"].iloc[0] == old["fraud_probability"].iloc[0]
    assert new["threshold_used"].iloc[0] == .1


def test_explanation_covers_thirteen_features():
    result = explain_transaction(routine())
    assert len(result) == 13
    assert np.isfinite(result["log_odds_contribution"]).all()


def test_global_importance_thirteen():
    scores = global_importance()
    assert len(scores) == 13
    assert scores["mean_split_gain"].ge(0).all()

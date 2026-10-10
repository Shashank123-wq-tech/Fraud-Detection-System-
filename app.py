"""Streamlit frontend for the uploaded Financial Fraud Detection project.

Run from repository root: streamlit run app.py
Uses the same selected model as the FastAPI app, exported to portable JSON.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.risk_engine import RiskThresholds
from fraud_ui.service import (
    MAX_BATCH_ROWS,
    ROOT,
    apply_risk_policy,
    explain_transaction,
    global_importance,
    load_fraud_model,
    screen_batch,
)

st.set_page_config(
    page_title="FraudScope | Transaction Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACTIONS = ["approve", "otp", "review", "block"]
COLORS = {
    "approve": "#35D0BA", "otp": "#F4BA60",
    "review": "#F08B67", "block": "#ED6078", "invalid": "#93A0B9",
}
THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#DFE7F3")

st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1350px; }
[data-testid="stSidebar"] { border-right: 1px solid #25354F; }
[data-testid="stMetric"] { padding: 15px 16px; background: #142238; border-radius: 14px; border: 1px solid #273B55; }
[data-testid="stMetricLabel"] { color: #AABBD1; }
.hero { background: linear-gradient(105deg, #142F3B 0%, #14233C 64%, #18203B 100%);
    padding: 25px 30px; border: 1px solid #294355; border-radius: 18px; margin-bottom: 20px; }
.hero .eyebrow { color: #35D0BA; letter-spacing: .17em; font-weight: 800; font-size: .75rem; }
.hero h1 { color: #EFF7FF; font-size: 2.2rem; margin: 4px 0 8px; line-height: 1.12; }
.hero p { color: #B6C7D9; font-size: 1.03rem; margin: 0; }
.muted { color: #A9B7C9; font-size: .91rem; }
.stButton > button[kind="primary"] { font-weight: 700; }
</style>
""", unsafe_allow_html=True)


def fmt_pct(number: float) -> str:
    return f"{100 * float(number):,.2f}%"


def plot(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(**THEME, margin=dict(l=15, r=15, t=35, b=15))
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def fraud_gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "valueformat": ".2f", "font": {"size": 41}},
        title={"text": "Estimated fraud probability", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#ABBBD1"},
            "bar": {"color": "#E3EBF7"},
            "bgcolor": "#18243A", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#175C5C"},
                {"range": [30, 60], "color": "#7A6230"},
                {"range": [60, 85], "color": "#89533C"},
                {"range": [85, 100], "color": "#7D3449"},
            ],
        },
    ))
    return fig


def sample_transaction(name: str) -> dict:
    samples = {
        "routine": dict(type="TRANSFER", amount=1000.0, oldbalanceOrg=5000.0,
                        newbalanceOrig=4000.0, oldbalanceDest=10000.0, newbalanceDest=11000.0),
        "account_drain": dict(type="CASH_OUT", amount=2500.0, oldbalanceOrg=2500.0,
                              newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=0.0),
    }
    return samples[name]


def populate_form(name: str) -> None:
    for key, value in sample_transaction(name).items():
        st.session_state[f"input_{key}"] = value


def csv_download_bytes(data: pd.DataFrame) -> bytes:
    """Quote and neutralize spreadsheet formulas in untrusted text cells."""
    prepared = data.copy()
    for col in prepared.select_dtypes(include=["object", "str", "string"]).columns:
        def clean(value):
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                return "'" + value
            return value
        prepared[col] = prepared[col].map(clean)
    return prepared.to_csv(index=False).encode("utf-8-sig")


def scored_rows() -> tuple[pd.DataFrame | None, str]:
    preferred = st.session_state.get("last_scored_kind")
    if preferred == "single" and "single_scored" in st.session_state:
        return st.session_state["single_scored"], "latest single transaction"
    if preferred == "batch" and "batch_scored" in st.session_state:
        return st.session_state["batch_scored"], "latest CSV batch"
    if "batch_scored" in st.session_state:
        return st.session_state["batch_scored"], "CSV batch"
    if "single_scored" in st.session_state:
        return st.session_state["single_scored"], "single transaction"
    return None, ""


def draw_overview(thresholds: RiskThresholds) -> None:
    st.subheader("Operations overview")
    raw, source = scored_rows()
    if raw is None:
        st.info("No transactions screened in this browser session. Use **Check Transaction** or **Batch Screening** to generate live results from your model.")
        st.markdown("#### What you can do")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**01 / Score a transaction**")
            st.caption("Enter six transaction fields to receive a fraud score and action.")
        with b:
            st.markdown("**02 / Screen a CSV file**")
            st.caption("Analyze up to 50,000 rows and export an enriched CSV.")
        with c:
            st.markdown("**03 / Inspect model behavior**")
            st.caption("Explore feature importance and per-transaction contributions.")
        return

    data = apply_risk_policy(raw, thresholds)
    valid = data.loc[data["fraud_probability"].notna()].copy()
    st.caption(f"Showing {source}. Results are held temporarily in server memory for your app session, not in a database.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions uploaded", f"{len(data):,}")
    m2.metric("Valid scores", f"{len(valid):,}")
    m3.metric("Flagged for review/block", f"{valid['is_fraud'].fillna(False).sum():,}")
    m4.metric("Mean fraud probability", fmt_pct(valid["fraud_probability"].mean()) if len(valid) else "N/A")
    if valid.empty:
        st.warning("No valid transactions were found. Correct the input errors and resubmit.")
        st.dataframe(data[["source_row", "validation_error"]], width="stretch", hide_index=True)
        return

    c1, c2 = st.columns(2)
    with c1:
        counts = valid["recommended_action"].value_counts().reindex(ACTIONS, fill_value=0)
        fig = px.pie(names=[name.upper() for name in counts.index], values=counts.values,
                     hole=.65, color=list(counts.index),
                     color_discrete_map={n: COLORS[n] for n in ACTIONS},
                     title="Risk action distribution")
        fig.update_layout(showlegend=True)
        plot(fig, 355)
    with c2:
        chart = valid.copy()
        chart["Probability (%)"] = chart["fraud_probability"] * 100
        fig = px.histogram(chart, x="Probability (%)", nbins=25,
                           title="Predicted probability distribution",
                           color_discrete_sequence=["#35D0BA"])
        fig.update_layout(yaxis_title="Number of transactions")
        plot(fig, 355)

    if "type" in valid.columns:
        bytype = pd.crosstab(valid["type"], valid["recommended_action"])
        fig = go.Figure()
        for action in ACTIONS:
            if action in bytype.columns:
                fig.add_bar(name=action.upper(), x=bytype.index, y=bytype[action], marker_color=COLORS[action])
        fig.update_layout(title="Risk decisions by transaction type", barmode="stack", xaxis_title="Transaction type", yaxis_title="Count")
        plot(fig, 340)
    st.markdown("#### Highest-risk transactions")
    columns = [c for c in ["source_row", "transaction_id", "type", "amount", "fraud_probability", "recommended_action"] if c in valid]
    st.dataframe(valid.sort_values("fraud_probability", ascending=False).head(20)[columns],
                 width="stretch", hide_index=True)


def draw_single(thresholds: RiskThresholds) -> None:
    st.subheader("Check a transaction")
    st.caption("Use PaySim-style input fields. The model was trained **only** for TRANSFER and CASH_OUT transactions.")
    b1, b2, space = st.columns([1.45, 1.55, 3])
    with b1:
        st.button("↺ Routine example", on_click=populate_form, args=("routine",), width="stretch")
    with b2:
        st.button("↺ Account-drain example", on_click=populate_form, args=("account_drain",), width="stretch")

    if "input_type" not in st.session_state:
        populate_form("routine")
    with st.form("single_form"):
        a, b = st.columns(2, gap="large")
        with a:
            tx_type = st.selectbox("Transaction type", ("TRANSFER", "CASH_OUT"), key="input_type")
            amount = st.number_input("Transaction amount", min_value=0.01, max_value=1e12, step=100.0,
                                     format="%.2f", key="input_amount")
            sender_before = st.number_input("Sender balance · before", min_value=0.0, max_value=1e12,
                                            step=100.0, format="%.2f", key="input_oldbalanceOrg")
        with b:
            sender_after = st.number_input("Sender balance · after", min_value=0.0, max_value=1e12,
                                           step=100.0, format="%.2f", key="input_newbalanceOrig")
            receiver_before = st.number_input("Receiver balance · before", min_value=0.0, max_value=1e12,
                                              step=100.0, format="%.2f", key="input_oldbalanceDest")
            receiver_after = st.number_input("Receiver balance · after", min_value=0.0, max_value=1e12,
                                             step=100.0, format="%.2f", key="input_newbalanceDest")
        with st.expander("Optional reference metadata"):
            tx_id = st.text_input("Transaction reference (not used for prediction)", max_chars=100, placeholder="e.g. TXN-1001")
        sent = st.form_submit_button("Analyze transaction", type="primary", width="stretch")

    if sent:
        payload = dict(type=tx_type, amount=amount, oldbalanceOrg=sender_before,
                       newbalanceOrig=sender_after, oldbalanceDest=receiver_before,
                       newbalanceDest=receiver_after, transaction_id=tx_id or "manual-transaction")
        try:
            result = screen_batch(pd.DataFrame([payload]), thresholds)
            if result["record_status"].iloc[0] == "invalid":
                st.error(str(result["validation_error"].iloc[0]))
            else:
                st.session_state["single_scored"] = result
                st.session_state["single_payload"] = payload
                st.session_state["last_scored_kind"] = "single"
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Could not score transaction: {exc}")

    if "single_scored" not in st.session_state:
        return
    last = apply_risk_policy(st.session_state["single_scored"], thresholds).iloc[0]
    if pd.isna(last["fraud_probability"]):
        return
    prob = float(last["fraud_probability"])
    action = str(last["recommended_action"])
    st.divider()
    st.markdown("### Prediction result")
    left, right = st.columns([1, 1.2])
    with left:
        plot(fraud_gauge(prob), 310)
    with right:
        st.metric("Recommended action", action.upper())
        st.metric("Fraud decision", "FRAUD FLAGGED" if last["is_fraud"] else "NOT FLAGGED")
        st.metric("Flagging threshold", fmt_pct(thresholds.review))
        st.caption("Fraud probabilities are model scores, not guaranteed calibrated risks. Action thresholds are illustrative policies, not validated operational cutoffs.")

    action_messages = {
        "approve": "Approve under the currently selected demo policy.",
        "otp": "Request additional verification under the demo policy.",
        "review": "Send for manual investigation under the demo policy.",
        "block": "Block under the demo policy. Do not automate real-world payment blocks from this demo.",
    }
    st.info(action_messages[action])
    with st.expander("Why did the model assign this score? · Local feature contributions", expanded=True):
        try:
            expl = explain_transaction(st.session_state["single_payload"])
            ranked = expl.head(10).sort_values("log_odds_contribution")
            fig = px.bar(ranked, x="log_odds_contribution", y="feature", orientation="h",
                         color="log_odds_contribution", color_continuous_scale=["#35D0BA", "#AEBBD1", "#ED6078"],
                         color_continuous_midpoint=0, title="Main contributions to fraud log-odds")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Effect on model log-odds (not percentage points)", yaxis_title="")
            plot(fig, 360)
            st.caption("Positive values push the model toward fraud; negative values push it away. This is a local XGBoost attribution, not a causal explanation.")
            st.dataframe(expl, width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Feature explanation unavailable: {exc}")
    scored_record = apply_risk_policy(st.session_state["single_scored"], thresholds).iloc[0]
    export = {"transaction_id": str(scored_record.get("transaction_id", "")),
              "fraud_probability": prob, "is_fraud": bool(scored_record["is_fraud"]),
              "recommended_action": action, "threshold_used": thresholds.review,
              "model": "xgboost_fraud_model_streamlit.json"}
    st.download_button("Download prediction (JSON)", data=json.dumps(export, indent=2),
                       file_name="fraud_prediction.json", mime="application/json")


def draw_batch(thresholds: RiskThresholds) -> None:
    st.subheader("Batch transaction screening")
    st.caption(f"Upload a PaySim-style CSV with up to {MAX_BATCH_ROWS:,} records. Only valid rows are scored; invalid rows are retained and marked with errors.")
    with st.expander("CSV schema · mandatory columns"):
        st.code("type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest", language="text")
        st.caption("Allowed types: TRANSFER, CASH_OUT. Amount must be positive; balances non-negative. An optional transaction_id column is preserved.")
    sample_path = ROOT / "examples" / "sample_transactions.csv"
    st.download_button("Download sample CSV", data=sample_path.read_bytes(),
                       file_name="sample_transactions.csv", mime="text/csv")
    upload = st.file_uploader("Upload transaction CSV", type=["csv"],
                              help="Only CSV files. No server-side database or retained upload storage is used.")
    if upload is not None:
        try:
            if upload.size > 20 * 1024 * 1024:
                raise ValueError("Maximum CSV size: 20 MB.")
            df = pd.read_csv(upload, low_memory=False, dtype={"transaction_id": "string"}, on_bad_lines="error")
            if df.empty:
                raise ValueError("The CSV has no rows.")
            st.caption(f"Detected {len(df):,} rows and {len(df.columns)} columns.")
            st.dataframe(df.head(10), width="stretch", hide_index=True)
            if st.button("Screen uploaded transactions", type="primary", width="stretch"):
                with st.spinner("Scoring valid transactions using the trained XGBoost model..."):
                    st.session_state["batch_scored"] = screen_batch(df, thresholds)
                    st.session_state["last_scored_kind"] = "batch"
                st.success("Batch scoring complete.")
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError, RuntimeError) as exc:
            st.error(f"Unable to process CSV: {exc}")

    if "batch_scored" not in st.session_state:
        return
    results = apply_risk_policy(st.session_state["batch_scored"], thresholds)
    st.divider()
    st.markdown("### Screening results")
    total = len(results)
    valid_count = int(results["fraud_probability"].notna().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uploaded", f"{total:,}")
    c2.metric("Scored", f"{valid_count:,}")
    c3.metric("Flagged", f"{int((results['recommended_action'].isin(['review', 'block'])).sum()):,}")
    c4.metric("Invalid", f"{total-valid_count:,}")
    options = ["all", "approve", "otp", "review", "block", "invalid"]
    selected = st.selectbox("Filter by outcome", options, key="batch_action_filter")
    subset = results if selected == "all" else results.loc[results["recommended_action"] == selected]
    st.dataframe(subset.head(5_000), width="stretch", hide_index=True)
    if len(subset) > 5_000:
        st.caption("Showing first 5,000 filtered rows; the download contains the entire batch.")
    st.download_button("Download all scored results (CSV)", data=csv_download_bytes(results),
                       file_name="fraud_screening_results.csv", mime="text/csv", type="primary")

    if "isFraud" in results.columns and valid_count:
        st.markdown("#### Optional ground-truth evaluation")
        st.caption("If uploaded isFraud labels are genuine holdout labels (0/1), the following evaluates those rows only. Do not treat training data results as an unbiased evaluation.")
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        evaluated = results.loc[results["fraud_probability"].notna()].copy()
        y = pd.to_numeric(evaluated["isFraud"], errors="coerce")
        mask = y.isin([0, 1])
        if mask.any():
            y_true = y.loc[mask].astype(int)
            y_pred = evaluated.loc[mask, "is_fraud"].astype(int)
            a, b, c = st.columns(3)
            a.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
            b.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
            if y_true.nunique() == 2:
                c.metric("PR-AUC (average precision)", f"{average_precision_score(y_true, evaluated.loc[mask, 'fraud_probability']):.3f}")
            else:
                c.metric("PR-AUC", "N/A (single class)")
            st.caption(f"Evaluated {len(y_true):,} valid labeled transactions.")


def draw_insights() -> None:
    st.subheader("Model intelligence")
    st.caption("Artifact: original FastAPI XGBoost model exported as a portable JSON model. Only TRANSFER and CASH_OUT are within training scope.")
    a, b, c = st.columns(3)
    a.metric("Model family", "XGBoost")
    b.metric("Input features", "13")
    c.metric("Supported classes", "2 transaction types")
    st.markdown("#### Global feature importance")
    importance = global_importance()
    fig = px.bar(importance.sort_values("mean_split_gain"), x="mean_split_gain", y="feature",
                 orientation="h", title="XGBoost mean split gain", color_discrete_sequence=["#35D0BA"])
    fig.update_layout(xaxis_title="Average split gain", yaxis_title="")
    plot(fig, 480)
    st.caption("Gain indicates feature use in tree splits; it does not establish a causal relationship or imply that changing a feature will prevent fraud.")
    with st.expander("Feature schema and transformations"):
        st.dataframe(importance, width="stretch", hide_index=True)
        st.markdown("The app calls the original `src/features.py::engineer_features()` function, guaranteeing the same 13 inputs and order used by training and FastAPI.")
    st.markdown("#### Evaluation integrity")
    st.warning("The uploaded README and model card report different performance figures. Their figures were not independently reproducible from the upload because the held-out labeled evaluation dataset was not supplied. Do not publish either set as newly verified performance.")
    metrics = pd.DataFrame([
        {"Source": "README model comparison", "Reported PR-AUC": 0.9979, "Reported recall": 0.9970},
        {"Source": "Model card", "Reported PR-AUC": 1.0, "Reported recall": 0.9998},
    ])
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.markdown("#### Important limitations")
    st.markdown("""
    - This is a **demonstration trained on synthetic PaySim data**, not a production-certified fraud control.
    - Post-transaction balance fields (`newbalanceOrig` and `newbalanceDest`) are required and may **not be available at the moment of authorization**. Do not claim genuine pre-authorization detection unless feature availability is validated.
    - Model scores have not been shown to be calibrated probabilities on live payment traffic.
    - Thresholds are examples; a real platform needs cost-based calibration, manual review workflows, audit logs, privacy controls, monitoring, and periodic retraining.
    """)


def main() -> None:
    st.sidebar.markdown("## 🛡️ FraudScope")
    st.sidebar.caption("Financial transaction risk intelligence")
    page = st.sidebar.radio("Workspace", ["Overview", "Check Transaction", "Batch Screening", "Model Insights"],
                            label_visibility="collapsed")
    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Risk policy · interactive", expanded=False):
        otp = st.slider("OTP threshold", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
        review = st.slider("Review / fraud threshold", min_value=0.05, max_value=0.95, value=0.60, step=0.05)
        block = st.slider("Block threshold", min_value=0.05, max_value=0.95, value=0.85, step=0.05)
        st.caption("Changes risk decisions, **not** the model output. Thresholds are not verified for live payment operations.")
    if not (0 < otp < review < block < 1):
        st.sidebar.error("Thresholds must satisfy OTP < Review < Block.")
        st.stop()
    thresholds = RiskThresholds(otp=otp, review=review, block=block)

    try:
        model = load_fraud_model()
        if model is None:
            raise RuntimeError("Model unavailable")
        st.sidebar.success("● XGBoost model ready")
    except Exception as exc:
        st.error(f"Model startup failed: {exc}")
        st.info("Confirm `models/xgboost_fraud_model_streamlit.json` and `models/feature_columns.json` are included in the deployment repository.")
        st.stop()
    if st.sidebar.button("Clear session results", width="stretch"):
        for key in ["single_scored", "single_payload", "batch_scored", "last_scored_kind"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.caption("No external API, credentials, or training data required.")

    st.markdown("""<div class="hero">
      <div class="eyebrow">FINANCIAL RISK LAB / ML POWERED</div>
      <h1>FraudScope · Transaction Intelligence</h1>
      <p>Analyze individual payments, investigate fraud drivers, and screen transaction files with your trained XGBoost model.</p>
    </div>""", unsafe_allow_html=True)
    if page == "Overview":
        draw_overview(thresholds)
    elif page == "Check Transaction":
        draw_single(thresholds)
    elif page == "Batch Screening":
        draw_batch(thresholds)
    else:
        draw_insights()
    st.divider()
    st.caption("Educational research prototype • PaySim synthetic data • Never upload real customer financial data to a public demo • No real payments are approved or blocked by this interface")


if __name__ == "__main__":
    main()

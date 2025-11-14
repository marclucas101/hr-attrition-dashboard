# App.py – HR Attrition Dashboard (FINAL VERSION)

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", page_icon="📉")
st.title("HR Attrition Risk Dashboard")

SAMPLE_FILE = pathlib.Path(__file__).parent / "hr_attrition_scored.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_default_data():
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    st.error("Default sample file not found. Please upload a CSV.")
    st.stop()


def ensure_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure AttritionFlag, AttritionRisk, RiskTier, Action exist.
    If missing, train risk model and predict.
    """
    df = df.copy()

    # Determine if dataset has attrition column
    has_target = "Attrition" in df.columns or "AttritionFlag" in df.columns

    # Case 1 — Already scored: just normalize
    if "AttritionRisk" in df.columns:
        df["AttritionRisk"] = df["AttritionRisk"].astype(float)
        df["AttritionFlag"] = np.where(df["AttritionRisk"] >= 0.5, 1, 0)
    else:
        # Case 2 — Has Attrition labels → Train and score on full dataset
        if has_target:
            if "AttritionFlag" not in df.columns:
                df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

            model, X_proc, y = train_model(df)
            df = add_predictions_to_df(df, model, X_proc)

        # Case 3 — No target, no risk → Cannot train → Assume low risk but keep UI working
        else:
            df["AttritionFlag"] = 0
            df["AttritionRisk"] = 0.01

    # Add RiskTier
    if "RiskTier" not in df.columns:
        df["RiskTier"] = df["AttritionRisk"].apply(
            lambda p: "High" if p >= 0.50 else "Medium" if p >= 0.30 else "Low"
        )

    # Add recommended actions
    if "Action" not in df.columns:
        df["Action"] = df["RiskTier"].map(
            {
                "High": "Immediate stay interview & retention plan",
                "Medium": "Manager check-in within 1 month",
                "Low": "Maintain engagement",
            }
        )

    return df


# -----------------------------------------------------------------------------
# Model Training + Prediction
# -----------------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    drop_cols = [
        "Attrition", "AttritionFlag", "AttritionRisk", "RiskTier", "Action",
        "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["AttritionFlag"]

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X, y)

    return pipe, X, y


def add_predictions_to_df(df, model, X_proc):
    """Predict AttritionFlag + risk probability"""
    preds = model.predict(model.named_steps["pre"].transform(X_proc))
    proba = model.predict_proba(model.named_steps["pre"].transform(X_proc))[:, 1]

    df["AttritionFlag"] = preds
    df["AttritionRisk"] = proba
    return df


# -----------------------------------------------------------------------------
# Feature Importance
# -----------------------------------------------------------------------------
def get_feature_importance(model, X, y):
    perm = permutation_importance(model, model.named_steps["pre"].transform(X), y,
                                  n_repeats=5, random_state=42, n_jobs=-1)

    feature_names = model.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(perm.importances_mean, index=feature_names)

    return (
        importances.sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("1️⃣ Upload HR Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(df_raw):,} rows")
else:
    df_raw = load_default_data()
    st.sidebar.info("Using sample file.")

df = ensure_predictions(df_raw)

st.sidebar.header("2️⃣ High-Risk Filter")
high_threshold = st.sidebar.slider("Minimum risk %", 10, 90, 40, 5) / 100


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_summary, tab_risk, tab_importance = st.tabs(
    ["📊 Historic Data", "🚨 High-Risk Employees", "🎯 Feature Importance"]
)

# -----------------------------------------------------------------------------
# TAB 1: Summary
# -----------------------------------------------------------------------------
with tab_summary:
    st.subheader("Historic Attrition Summary")

    if "Department" in df.columns:
        summary = df.groupby("Department").agg(
            Employees=("AttritionFlag", "size"),
            AttritionRate=("AttritionFlag", lambda x: (x.mean()) * 100),
            AvgRisk=("AttritionRisk", lambda x: (x.mean()) * 100),
            HighRisk=("RiskTier", lambda s: (s == "High").sum()),
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Attrition Rate", f"{summary['AttritionRate'].mean():.1f}%")
        c2.metric("Avg Risk Score", f"{summary['AvgRisk'].mean():.1f}%")
        c3.metric("High-Risk Employees", f"{(df['RiskTier']=='High').sum():,}")

        st.dataframe(summary.round(1), use_container_width=True)

        fig = px.bar(summary.round(1), x=summary.index, y="AvgRisk",
                     color="AvgRisk", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset has no Department column.")


# -----------------------------------------------------------------------------
# TAB 2: High-Risk List
# -----------------------------------------------------------------------------
with tab_risk:
    st.subheader("🚨 High-Risk Employees")

    at_risk = df[df["AttritionRisk"] >= high_threshold].copy()
    at_risk["AttritionRisk"] = (at_risk["AttritionRisk"] * 100).round(1)

    if at_risk.empty:
        st.info("No employees above selected risk threshold.")
    else:
        st.dataframe(at_risk, use_container_width=True, hide_index=True)
        st.download_button("Download CSV",
                           at_risk.to_csv(index=False).encode("utf-8"),
                           "high_risk_employees.csv",
                           mime="text/csv")


# -----------------------------------------------------------------------------
# TAB 3: Feature Importance
# -----------------------------------------------------------------------------
with tab_importance:
    st.subheader("🎯 Key Drivers of Attrition")

    if "Attrition" in df.columns or "AttritionFlag" in df.columns:
        model, X, y = train_model(df)
        imp = get_feature_importance(model, X, y)

        fig = px.bar(
            imp.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp)
    else:
        st.info("Dataset does not contain labels → drivers unavailable.")

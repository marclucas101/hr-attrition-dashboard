# App.py  – HR Attrition Dashboard

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("HR Attrition Risk Dashboard")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
SAMPLE_FILE = pathlib.Path(__file__).parent / "hr_attrition_scored.csv"


def load_default_data() -> pd.DataFrame:
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    st.error("Default file not found. Please upload a file.")
    st.stop()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target detection
    target_candidates = {
        "AttritionFlag": "binary",
        "Attrition": {"Yes": 1, "No": 0, "Y": 1, "N": 0},
        "Leave": {"Yes": 1, "No": 0},
        "Exited": {"Yes": 1, "No": 0},
        "left": {"1": 1, "0": 0},
    }

    found = False
    for col, mapping in target_candidates.items():
        if col in df.columns:
            if mapping == "binary":
                df["AttritionFlag"] = df[col].astype(int)
            else:
                df["AttritionFlag"] = (
                    df[col].astype(str).map(mapping).fillna(0).astype(int)
                )
            found = True
            break

    if not found:
        df["AttritionFlag"] = 0  # fallback - no attrition column found

    # Risk column detection
    risk_candidates = ["AttritionRisk", "risk", "RiskScore", "Probability", "score"]
    for col in risk_candidates:
        if col in df.columns:
            df["AttritionRisk"] = df[col].astype(float)
            break
    else:
        df["AttritionRisk"] = df["AttritionFlag"].astype(float)

    df["AttritionRisk"] = df["AttritionRisk"].clip(0, 1)

    # Risk Tier
    def get_tier(r):
        if r >= 0.5: return "High"
        if r >= 0.3: return "Medium"
        return "Low"
    df["RiskTier"] = df["AttritionRisk"].apply(get_tier)

    # Suggested actions
    df["Action"] = df["RiskTier"].map({
        "High": "Immediate stay interview",
        "Medium": "Manager follow-up",
        "Low": "Maintain engagement"
    })

    return df


def department_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Department" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("Department")
        .agg(
            Employees=("AttritionFlag", "size"),
            AttritionRate=("AttritionFlag", "mean"),
            AvgRisk=("AttritionRisk", "mean"),
            HighRisk=("RiskTier", lambda x: (x == "High").sum())
        )
        .reset_index()
    )
    summary["AttritionRate"] *= 100
    summary["AvgRisk"] *= 100
    return summary


def build_feature_importance(df: pd.DataFrame):
    if df["AttritionFlag"].nunique() < 2:
        return None

    drop_cols = [
        "AttritionFlag", "Attrition", "AttritionRisk",
        "RiskTier", "Action", "EmployeeNumber",
        "StandardHours", "Over18", "EmployeeCount"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["AttritionFlag"]

    if X.shape[1] == 0:
        return None

    num_cols = X.select_dtypes(include=["number","bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X, y)

    perm = permutation_importance(pipe, X, y, n_repeats=5, random_state=42)

    names = pipe.named_steps["pre"].get_feature_names_out()
    result = (
        pd.DataFrame({"Feature": names, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False)
        .head(20)
    )
    return result


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("1️⃣ Upload Data")
uploaded = st.sidebar.file_uploader(
    "Upload HR CSV file", type=["csv"]
)

df_raw = pd.read_csv(uploaded) if uploaded else load_default_data()
df = normalise_columns(df_raw)

st.sidebar.header("2️⃣ High-risk employee threshold")
high_threshold = st.sidebar.slider(
    "Minimum risk (%)",
    10, 90, 40, step=5
)
risk_cut = high_threshold / 100


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_dept, tab_risk, tab_imp = st.tabs([
    "📈 Department Summary",
    "⚠️ High-Risk Employees",
    "🧠 Feature Importance"
])

# TAB — Department
with tab_dept:
    st.subheader("Attrition Summary by Department")
    summary = department_summary(df)

    if summary.empty:
        st.info("No Department column found.")
    else:
        st.dataframe(summary, use_container_width=True)

        fig = px.bar(summary, x="Department", y="AvgRisk",
                     color="HighRisk", title="Avg Risk by Department")
        st.plotly_chart(fig, use_container_width=True)

# TAB — High-Risk Employees
with tab_risk:
    st.subheader("High-Risk Employees")
    high_df = df[df["AttritionRisk"] >= risk_cut].copy()

    if high_df.empty:
        st.info("No employees above threshold.")
    else:
        high_df["AttritionRisk"] = (high_df["AttritionRisk"] * 100).round(1)
        st.dataframe(high_df, use_container_width=True)

        st.download_button(
            "Download CSV", high_df.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_employees.csv"
        )

# TAB — Feature Importance
with tab_imp:
    st.subheader("What Drives Attrition?")
    result = build_feature_importance(df)

    if result is None:
        st.info("Not enough attrition variation for model training.")
    else:
        fig = px.bar(result, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result, use_container_width=True)


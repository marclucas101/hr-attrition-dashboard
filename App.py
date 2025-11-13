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
    page_icon="T_T",
)

st.title("HR Attrition Risk Dashboard")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
SAMPLE_FILE = pathlib.Path(__file__).parent / "hr_attrition_scored.csv"


def load_default_data() -> pd.DataFrame:
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    st.error("Sample file hr_attrition_scored.csv not found in the repo. Please upload a CSV.")
    st.stop()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise target / risk columns so the rest of the app can rely on them."""
    df = df.copy()

    # 1) Target column
    if "AttritionFlag" in df.columns:
        df["AttritionFlag"] = df["AttritionFlag"].astype(int)
    elif "Attrition" in df.columns:
        # Expect Yes / No
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    else:
        # If no target, create dummy column of zeros so aggregations still run
        df["AttritionFlag"] = 0

    # 2) Risk column
    risk_col = None
    for c in ["AttritionRisk", "attrition_risk", "RiskScore", "score"]:
        if c in df.columns:
            risk_col = c
            break

    if risk_col is None:
        # Fall back: if there is no explicit probability, use AttritionFlag as a proxy
        df["AttritionRisk"] = df["AttritionFlag"].astype(float)
        risk_col = "AttritionRisk"
    else:
        # Ensure float
        df[risk_col] = df[risk_col].astype(float)
        if risk_col != "AttritionRisk":
            df.rename(columns={risk_col: "AttritionRisk"}, inplace=True)

    # 3) Risk tier
    if "RiskTier" not in df.columns:
        def tier(p):
            if p >= 0.5:
                return "High"
            if p >= 0.3:
                return "Medium"
            return "Low"

        df["RiskTier"] = df["AttritionRisk"].apply(tier)

    # 4) Action column
    if "Action" not in df.columns:
        def action_for_tier(t):
            if t == "High":
                return "Immediate stay interview & retention plan"
            if t == "Medium":
                return "Manager check-in within 1 month"
            return "Maintain engagement"

        df["Action"] = df["RiskTier"].apply(action_for_tier)

    return df


def department_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Department" not in df.columns:
        st.warning("Column 'Department' not found. Department summary will be empty.")
        return pd.DataFrame()

    summary = (
        df.groupby("Department")
        .agg(
            Employees=("AttritionFlag", "size"),
            AttritionRate=("AttritionFlag", "mean"),
            AvgRisk=("AttritionRisk", "mean"),
            HighRisk=("RiskTier", lambda s: (s == "High").sum()),
        )
        .reset_index()
    )

    summary["AttritionRate"] = (summary["AttritionRate"] * 100).round(1)
    summary["AvgRisk"] = (summary["AvgRisk"] * 100).round(1)
    return summary


def build_feature_importance(df: pd.DataFrame):
    """
    Train a simple RandomForest model on the uploaded data and compute
    permutation importance. This is deliberately light-weight and only
    meant to approximate drivers.
    """
    if "AttritionFlag" not in df.columns:
        return None

    # Drop obvious non-features / leakage columns
    drop_cols = [
        "AttritionFlag",
        "Attrition",
        "AttritionRisk",
        "RiskTier",
        "Action",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["AttritionFlag"].astype(int)

    if X.shape[1] == 0 or y.nunique() < 2:
        return None

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X, y)

    # permutation importance on the *pipeline*
    perm = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(perm.importances_mean, index=feature_names)

    top = (
        importances.sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )
    return top


# ---------------------------------------------------------
# Sidebar – data upload
# ---------------------------------------------------------
st.sidebar.header("1. Data")

uploaded = st.sidebar.file_uploader("Upload HR attrition CSV", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded file with {len(df_raw):,} rows.")
else:
    df_raw = load_default_data()
    st.sidebar.info("Using bundled sample file hr_attrition_scored.csv.")

df = normalise_columns(df_raw)

st.sidebar.header("2. High-risk filter")
high_threshold = st.sidebar.slider(
    "Minimum risk (%) for High-risk list",
    min_value=10,
    max_value=90,
    value=40,
    step=5,
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_dept, tab_risk, tab_importance = st.tabs(
    ["Historic Data", "High-Risk Employees", "Feature Importance"]
)

# ---------------------------------------------------------
# Tab 1 – History Attrition Data
# ---------------------------------------------------------
with tab_dept:
    st.subheader("History Attrition Data")

    summary = department_summary(df)
    if not summary.empty:
        c1, c2, c3 = st.columns(3)
        overall_attrition = df["AttritionFlag"].mean() * 100
        overall_risk = df["AttritionRisk"].mean() * 100
        high_count = (df["RiskTier"] == "High").sum()

        c1.metric("Overall attrition rate", f"{overall_attrition:.1f}%")
        c2.metric("Average risk score", f"{overall_risk:.1f}%")
        c3.metric("High-risk employees", f"{high_count:,}")

        st.dataframe(summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            summary,
            x="Department",
            y="AvgRisk",
            color="AvgRisk",
            color_continuous_scale="Reds",
            labels={"AvgRisk": "Average risk (%)"},
            title="Average predicted attrition risk by department",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'Department' column found, so this view is empty.")


# ---------------------------------------------------------
# Tab 2 – High-risk employees
# ---------------------------------------------------------
with tab_risk:
    st.subheader("Highest-Risk Employees")

    risk_cut = high_threshold / 100.0
    df_sorted = df.sort_values("AttritionRisk", ascending=False).copy()
    top_high = df_sorted[df_sorted["AttritionRisk"] >= risk_cut]

    if top_high.empty:
        st.info(
            f"No employees above {high_threshold}% risk in this dataset. "
            "Try lowering the slider on the left."
        )
    else:
        display_cols = [
            c
            for c in [
                "EmployeeNumber",
                "Department",
                "JobRole",
                "Age",
                "Gender",
                "MonthlyIncome",
                "AttritionRisk",
                "RiskTier",
                "Action",
            ]
            if c in top_high.columns
        ]

        table = top_high[display_cols].copy()
        table["AttritionRisk"] = (table["AttritionRisk"] * 100).round(1)

        st.write(
            f"Employees with predicted risk ≥ **{high_threshold}%** "
            f"({len(table):,} employees)."
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download high-risk list (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_employees.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# Tab 3 – Feature importance
# ---------------------------------------------------------
with tab_importance:
    st.subheader("What Drives Attrition? (Feature Importance)")
    st.caption(
        "This view trains a light-weight Random Forest model on the uploaded data "
        "and uses permutation importance to approximate key drivers."
    )

    importance_df = build_feature_importance(df)

    if importance_df is None:
        st.info(
            "Could not compute feature importance. Make sure your data contains a "
            "binary attrition indicator (e.g., 'AttritionFlag' or 'Attrition' Yes/No) "
            "and enough rows."
        )
    else:
        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Importance", "Feature": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True, hide_index=True)



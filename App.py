# app.py  – HR Attrition Risk Dashboard (with on-the-fly prediction)

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

BASE_DIR = pathlib.Path(__file__).parent
TRAIN_FILE = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SCORE_SAMPLE_FILE = BASE_DIR / "NO ATTRITION-WA_Fn-UseC_-HR-Employee-data.csv"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def load_training_data() -> pd.DataFrame:
    """Load labelled historical data with Attrition = Yes/No."""
    if TRAIN_FILE.exists():
        df = pd.read_csv(TRAIN_FILE)
        if "Attrition" not in df.columns:
            st.error(
                "Training file found but it has no 'Attrition' column. "
                "Please upload a valid labelled dataset."
            )
            st.stop()
        return df

    st.error(
        "Default training file 'WA_Fn-UseC_-HR-Employee-Attrition.csv' "
        "is not in the app folder. Upload a labelled training CSV in the sidebar."
    )
    st.stop()


def load_scoring_data(uploaded) -> pd.DataFrame:
    """Load current-employee data to score."""
    if uploaded is not None:
        return pd.read_csv(uploaded)

    if SCORE_SAMPLE_FILE.exists():
        st.sidebar.info(
            "No file uploaded. Using bundled sample "
            "'NO ATTRITION-WA_Fn-UseC_-HR-Employee-data.csv'."
        )
        return pd.read_csv(SCORE_SAMPLE_FILE)

    st.warning("Upload a 'current employees' CSV on the left to score attrition risk.")
    st.stop()


DROP_COLS = [
    "Attrition",
    "AttritionFlag",
    "AttritionRisk",
    "RiskTier",
    "Action",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def train_model(train_df: pd.DataFrame, score_df: pd.DataFrame):
    """
    Train a RandomForest on the labelled historical data.

    We only use features that exist in BOTH training and scoring datasets,
    so the model can be applied safely to the current employees file.
    """
    # Target
    if "Attrition" not in train_df.columns:
        st.error("Training data must contain an 'Attrition' column (Yes/No).")
        st.stop()

    y = train_df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

    # Base feature sets
    train_base = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    score_base = score_df.drop(columns=[c for c in DROP_COLS if c in score_df.columns])

    # Keep only common columns (to avoid problems like 'Worklife balance')
    common_cols = sorted(set(train_base.columns).intersection(score_base.columns))
    if len(common_cols) == 0:
        st.error(
            "No common feature columns found between training and scoring datasets. "
            "Check that they have the same HR fields."
        )
        st.stop()

    X_train = train_base[common_cols]

    # Numeric vs categorical
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("rf", rf)])
    pipe.fit(X_train, y)

    # Simple feature importance from the trained forest
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["rf"].feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(25)
    )

    return pipe, common_cols, importance_df, X_train


def score_employees(model: Pipeline, feature_cols, score_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the trained model to the current-employee data."""
    base = score_df.drop(columns=[c for c in DROP_COLS if c in score_df.columns])
    X_score = base.reindex(columns=feature_cols)

    proba = model.predict_proba(X_score)[:, 1]  # P(attrition = 1)

    df_scored = score_df.copy()
    df_scored["AttritionRisk"] = proba

    # AttritionFlag is 0 for current employees unless we actually have labels
    if "Attrition" in df_scored.columns:
        df_scored["AttritionFlag"] = (
            df_scored["Attrition"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        )
    else:
        df_scored["AttritionFlag"] = 0

    # Risk tier + recommended action
    def tier(p):
        if p >= 0.5:
            return "High"
        if p >= 0.3:
            return "Medium"
        return "Low"

    def action_for_tier(t):
        if t == "High":
            return "Immediate stay interview & retention plan"
        if t == "Medium":
            return "Manager check-in within 1 month"
        return "Maintain engagement"

    df_scored["RiskTier"] = df_scored["AttritionRisk"].apply(tier)
    df_scored["Action"] = df_scored["RiskTier"].apply(action_for_tier)

    return df_scored


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


# ---------------------------------------------------------
# Sidebar – data upload & training
# ---------------------------------------------------------
st.sidebar.header("1. Data")

scoring_file = st.sidebar.file_uploader(
    "Upload **current employees** HR CSV (to predict attrition risk)",
    type=["csv"],
)

# Load data
train_df = load_training_data()
score_raw = load_scoring_data(scoring_file)
st.sidebar.success(f"Loaded scoring file with {len(score_raw):,} rows.")

# Train model on historical labelled data & score current employees
model, feature_cols, importance_df, X_train_for_info = train_model(train_df, score_raw)
df = score_employees(model, feature_cols, score_raw)

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
    ["Department Summary", "High-Risk Employees", "Feature Importance"]
)

# ---------------------------------------------------------
# Tab 1 – Department summary
# ---------------------------------------------------------
with tab_dept:
    st.subheader("Department Summary")

    summary = department_summary(df)
    if not summary.empty:
        c1, c2, c3 = st.columns(3)
        overall_attrition = df["AttritionFlag"].mean() * 100  # may be 0 for current staff
        overall_risk = df["AttritionRisk"].mean() * 100
        high_count = (df["RiskTier"] == "High").sum()

        c1.metric("Observed attrition rate", f"{overall_attrition:.1f}%")
        c2.metric("Average predicted risk", f"{overall_risk:.1f}%")
        c3.metric("High-risk employees", f"{high_count:,}")

        st.dataframe(summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            summary,
            x="Department",
            y="AvgRisk",
            color="AvgRisk",
            color_continuous_scale="Reds",
            labels={"AvgRisk": "Average predicted risk (%)"},
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
            f"No employees above {high_threshold}% predicted risk in this dataset. "
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
# Tab 3 – Feature importance (from Random Forest)
# ---------------------------------------------------------
with tab_importance:
    st.subheader("What Drives Attrition? (Feature Importance)")
    st.caption(
        "The model is trained on the historical labelled dataset "
        "('WA_Fn-UseC_-HR-Employee-Attrition.csv'). "
        "Feature importances come from the trained Gradient Boosting."
    )

    if importance_df is None or importance_df.empty:
        st.info(
            "Could not compute feature importance. "
            "Check that the training data is valid."
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


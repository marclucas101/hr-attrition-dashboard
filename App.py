# app.py  – HR Attrition Risk Dashboard (train on labelled data, score current employees)

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
# Paths
# ---------------------------------------------------------
HERE = pathlib.Path(__file__).parent

# Historical, labelled attrition data – used to TRAIN the model
TRAIN_FILE = HERE / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Optional sample scoring file – used only if user does not upload anything
SAMPLE_FILE = HERE / "hr_attrition_scored.csv"  # you can also swap this to NO ATTRITION file if you like


# ---------------------------------------------------------
# Model training
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_attrition_model():
    """
    Train a Random Forest model on historical labelled data.

    Returns
    -------
    model : Pipeline
        Preprocessing + classifier pipeline.
    feature_cols : list[str]
        Columns used as model inputs.
    importance_df : pd.DataFrame
        Top permutation importances for display.
    """
    if not TRAIN_FILE.exists():
        st.error(
            f"Training file not found: {TRAIN_FILE.name}. "
            "Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the same folder as app.py."
        )
        st.stop()

    df_train = pd.read_csv(TRAIN_FILE)

    # Create binary target
    if "Attrition" not in df_train.columns:
        st.error(
            "Training file must contain an 'Attrition' column with 'Yes'/'No' values."
        )
        st.stop()

    df_train["AttritionFlag"] = df_train["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    target_col = "AttritionFlag"

    # Drop obvious IDs / constants
    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    df_model = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])

    X = df_model.drop(columns=[target_col, "Attrition"])
    y = df_model[target_col]

    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = [c for c in X.columns if c not in num_features]

    feature_cols = X.columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline([("pre", preprocess), ("rf", rf)])
    model.fit(X, y)

    # Permutation importance on the training data
    perm = permutation_importance(
        model, X, y, n_repeats=5, random_state=42, n_jobs=-1
    )
    # Names after preprocessing
    feature_names = model.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(perm.importances_mean, index=feature_names)

    importance_df = (
        importances.sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )

    return model, feature_cols, importance_df


model, MODEL_FEATURE_COLS, importance_df = train_attrition_model()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_default_data() -> pd.DataFrame:
    """Load sample scoring data if the user does not upload a file."""
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    st.error(
        "No file uploaded and sample file hr_attrition_scored.csv was not found. "
        "Please upload a CSV on the left."
    )
    st.stop()


def score_and_normalise(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Use the trained model to predict attrition risk for the uploaded dataset
    and create the standard columns AttritionRisk, AttritionFlag, RiskTier, Action.
    """
    df = df_raw.copy()

    # --- Ensure required feature columns are present -------------------------
    missing = [c for c in MODEL_FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(
            "Uploaded file is missing required columns used by the model:\n"
            + ", ".join(missing)
        )
        st.stop()

    X_score = df[MODEL_FEATURE_COLS]
    proba = model.predict_proba(X_score)[:, 1]  # probability of attrition
    df["AttritionRisk"] = proba.astype(float)

    # If actual Attrition is present, keep it. Otherwise, use a predicted flag
    if "AttritionFlag" in df.columns:
        df["AttritionFlag"] = df["AttritionFlag"].astype(int)
    elif "Attrition" in df.columns:
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    else:
        # For "NO ATTRITION" current-employee dataset: we don't know who left,
        # so treat anyone above 50% risk as a "predicted leaver"
        df["AttritionFlag"] = (df["AttritionRisk"] >= 0.5).astype(int)

    # Risk tiers based on probability
    if "RiskTier" not in df.columns:
        def tier(p):
            if p >= 0.5:
                return "High"
            if p >= 0.3:
                return "Medium"
            return "Low"

        df["RiskTier"] = df["AttritionRisk"].apply(tier)

    # Recommended action per tier
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
    """
    Aggregate predictions by Department.

    AttritionRate is based on AttritionFlag, which for unlabeled data is a
    predicted flag (risk >= 50%).
    """
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

df = score_and_normalise(df_raw)

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
        # Overall predicted attrition rate based on AttritionFlag
        overall_attrition = df["AttritionFlag"].mean() * 100
        overall_risk = df["AttritionRisk"].mean() * 100
        high_count = (df["RiskTier"] == "High").sum()

        c1.metric("Predicted attrition rate", f"{overall_attrition:.1f}%")
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
# Tab 3 – Feature importance (from training data)
# ---------------------------------------------------------
with tab_importance:
    st.subheader("What Drives Attrition? (Feature Importance)")
    st.caption(
        "These importances are computed from the Random Forest trained on the "
        "historical labelled dataset WA_Fn-UseC_-HR-Employee-Attrition.csv "
        "using permutation importance."
    )

    if importance_df is None or importance_df.empty:
        st.info(
            "Could not compute feature importance from the training data. "
            "Check that the training CSV is present and has an 'Attrition' column."
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

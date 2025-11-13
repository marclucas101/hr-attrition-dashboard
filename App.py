# ----------------------------
# HR ATTRITION DASHBOARD v3
# Fully fixed version
# ----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------
# Page Layout
# ---------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Dashboard",
    layout="wide"
)

st.title("HR Attrition Prediction Dashboard")

st.write("Upload a CSV file in IBM HR Attrition format to begin.")

# ---------------------------------------------------------
# File Upload Section
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload your HR dataset", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded)

# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------
required_cols = ["Attrition"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required column(s): {missing}")
    st.stop()

# ---------------------------------------------------------
# Target variable
# ---------------------------------------------------------
df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

y = df["AttritionFlag"]
X = df.drop(columns=["Attrition", "AttritionFlag"])

# Identify features
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = [c for c in X.columns if c not in num_features]

# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

base_model = HistGradientBoostingClassifier(random_state=42)

clf = Pipeline(steps=[
    ("pre", preprocess),
    ("cal", CalibratedClassifierCV(base_model, cv=3))
])

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# Predictions on full dataset
# ---------------------------------------------------------
df["AttritionRisk"] = clf.predict_proba(X)[:, 1]

def bucket(r):
    if r >= 0.40:
        return "High"
    elif r >= 0.20:
        return "Medium"
    return "Low"

df["RiskTier"] = df["AttritionRisk"].apply(bucket)

# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Department Summary", "High-Risk Employees", "Feature Importance"])

# =========================================================
# TAB 1 — DEPARTMENT SUMMARY
# =========================================================
with tab1:
    st.header("Department Risk Summary")

    if "Department" not in df.columns:
        st.warning("'Department' column not found. Cannot display department risk chart.")
    else:
        dept_summary = (
            df.groupby("Department")
            .agg(
                Avg_Risk=("AttritionRisk", "mean"),
                High_Risk_Count=("RiskTier", lambda s: (s == "High").sum()),
                Employees=("AttritionRisk", "count")
            )
            .reset_index()
        )

        dept_summary["Avg_Risk"] = dept_summary["Avg_Risk"] * 100

        fig = px.bar(
            dept_summary,
            x="Department",
            y="Avg_Risk",
            color="Avg_Risk",
            color_continuous_scale="Reds",
            title="Average Attrition Risk (%) by Department"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — HIGH-RISK EMPLOYEES
# =========================================================
with tab2:
    st.header("Highest-Risk Employees (Top 20)")

    display_cols = [
        c for c in ["EmployeeNumber", "Department", "JobRole",
                    "MonthlyIncome", "WorkLifeBalance",
                    "AttritionRisk", "RiskTier"]
        if c in df.columns
    ]

    if not display_cols:
        st.warning("No standard HR columns found to display.")
    else:
        top = df.sort_values("AttritionRisk", ascending=False).head(20)

        top["AttritionRisk"] = (top["AttritionRisk"] * 100).round(1)

        st.dataframe(top[display_cols])

# =========================================================
# TAB 3 — FIXED FEATURE IMPORTANCE
# =========================================================
with tab3:
    st.header("What Drives Attrition? (Feature Importance)")

    st.write("This uses Permutation Importance on the trained model.")

    # Safe Permutation Importance
    try:
        perm = permutation_importance(
            clf, X_test, y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        # Extract transformed feature names
        ohe = clf.named_steps["pre"].transformers_[1][1]
        ohe_names = ohe.get_feature_names_out(cat_features)

        feature_names = num_features + list(ohe_names)

        # FIX: Align lengths
        valid_len = min(len(feature_names), len(perm.importances_mean))

        importance_df = pd.DataFrame({
            "Feature": feature_names[:valid_len],
            "Importance": perm.importances_mean[:valid_len]
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            importance_df.head(20),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Drivers of Attrition",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error("Could not compute feature importance for this dataset.")
        st.exception(e)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import plotly.express as px

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# ----------------------------------------------------
# 1. FILE UPLOAD
# ----------------------------------------------------
st.sidebar.header("Upload HR Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Upload a CSV file to begin.")
    st.stop()

# ----------------------------------------------------
# 2. DETECT IF THIS IS A SCORED FILE
# ----------------------------------------------------
is_scored_file = all(col in df.columns for col in ["AttritionRisk", "RiskTier", "Action"])

# ----------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ----------------------------------------------------
page = st.sidebar.radio("Navigation", ["Department Summary", "High-Risk Employees", "Feature Importance"])

# ----------------------------------------------------
# 4. SHOW DEPARTMENT SUMMARY
# ----------------------------------------------------
if page == "Department Summary":

    if not is_scored_file:
        st.error("This page requires a scored CSV (with AttritionRisk, RiskTier, Action).")
        st.stop()

    st.title("Department Summary")

    dept_summary = (
        df.groupby("Department")
        .agg(
            n_employees=("EmployeeNumber", "count"),
            avg_risk=("AttritionRisk", "mean"),
            high_risk_count=("RiskTier", lambda x: (x == "High").sum())
        ).reset_index()
    )

    fig = px.bar(
        dept_summary,
        x="Department",
        y="avg_risk",
        color="avg_risk",
        color_continuous_scale="Reds",
        title="Average Predicted Attrition Risk by Department"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dept_summary)

# ----------------------------------------------------
# 5. SHOW HIGH-RISK EMPLOYEE LIST
# ----------------------------------------------------
elif page == "High-Risk Employees":

    if not is_scored_file:
        st.error("This page requires a scored CSV (with AttritionRisk, RiskTier, Action).")
        st.stop()

    st.title("Highest-Risk Employees")

    top = (
        df.sort_values("AttritionRisk", ascending=False)
        .head(20)[["EmployeeNumber", "Department", "JobRole", "AttritionRisk", "RiskTier", "Action"]]
    )

    st.dataframe(top)

# ----------------------------------------------------
# 6. FEATURE IMPORTANCE SECTION
# ----------------------------------------------------
elif page == "Feature Importance":

    st.title("What Drives Attrition? (Feature Importance)")

    # If scored file → disable section
    if is_scored_file:
        st.info("""
        ### ℹ️ Feature Importance Unavailable
        Your uploaded file is a **scored output file**, not raw HRIS data.

        To compute feature importance, upload a **raw HR dataset** that includes
        all employee features used for model training (e.g., BusinessTravel, JobRole, Age, MonthlyIncome, etc.).
        """)
        st.stop()

    # ----------------------------------------------------
    # RUN MODEL ON RAW DATA
    # ----------------------------------------------------
    df = df.copy()
    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    drop_cols = ["EmployeeCount", "StandardHours", "Over18"]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df_model.drop(columns=["Attrition", "AttritionFlag"])
    y = df_model["AttritionFlag"]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    base = HistGradientBoostingClassifier(max_depth=None, random_state=42)
    clf = Pipeline([
        ("pre", preprocess),
        ("cal", CalibratedClassifierCV(base, method="sigmoid", cv=3))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    # ----------------------------------------------------
    # PERMUTATION IMPORTANCE
    # ----------------------------------------------------
    st.subheader("Top Drivers of Attrition")

    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )

    # Get feature names
    ohe = clf.named_steps["pre"].transformers_[1][1]
    ohe_names = ohe.get_feature_names_out(cat_cols)

    feature_names = list(num_cols) + list(ohe_names)

    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False).head(20)

    fig = px.bar(
        importances,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

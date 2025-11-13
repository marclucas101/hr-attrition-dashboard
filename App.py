import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# ------------------------------------
# Page config
# ------------------------------------
st.set_page_config(page_title="HR Attrition Dashboard",
                   layout="wide",
                   page_icon="🖕")

st.title("HR Attrition Risk Dashboard")
st.write("Upload your HR dataset to begin.")

# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader("Upload HR dataset (.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# ------------------------------------
# Preprocess
# ------------------------------------
df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

drop_cols = ['EmployeeCount', 'Over18', 'StandardHours']
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df_model.drop(columns=["Attrition", "AttritionFlag"])
y = df_model["AttritionFlag"]

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = [c for c in X.columns if c not in num_features]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ------------------------------------
# Model
# ------------------------------------
base_clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)

clf = Pipeline(steps=[
    ("pre", preprocess),
    ("cal", CalibratedClassifierCV(base_clf, cv=3))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
# ------------------------------------
# Predict full dataset
# ------------------------------------
df["AttritionRisk"] = clf.predict_proba(X)[:, 1]

def risk_bucket(x):
    if x >= 0.40:
        return "High"
    elif x >= 0.20:
        return "Medium"
    else:
        return "Low"

df["RiskTier"] = df["AttritionRisk"].apply(risk_bucket)

# ------------------------------------
# Dashboard Layout
# ------------------------------------
st.header("Department-Level Risk Overview")

dept_summary = df.groupby("Department").agg(
    avg_risk=("AttritionRisk", "mean"),
    high_risk=("RiskTier", lambda s: (s == "High").sum()),
    headcount=("EmployeeNumber", "count")
).reset_index()

dept_summary["avg_risk"] = dept_summary["avg_risk"] * 100

fig = px.bar(
    dept_summary,
    x="Department",
    y="avg_risk",
    color="avg_risk",
    color_continuous_scale="Reds",
    title="Average Attrition Risk by Department (%)"
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# High Risk Employees
# ------------------------------------
st.header("Highest-Risk Employees (Top 20)")

top_risk = df.sort_values("AttritionRisk", ascending=False).head(20)
top_risk_display = top_risk[[
    "EmployeeNumber", "Department", "JobRole",
    "MonthlyIncome", "WorkLifeBalance",
    "AttritionRisk", "RiskTier"
]]

top_risk_display["AttritionRisk"] = (top_risk_display["AttritionRisk"] * 100).round(1)

st.dataframe(top_risk_display)

# ------------------------------------
# Feature Importance
# ------------------------------------
st.header("What Drives Attrition? (Feature Importance)")

with st.spinner("Computing feature importance..."):

    # --- Extract full feature names ---
    ohe = clf.named_steps["pre"].transformers_[1][1]
    ohe_features = list(ohe.get_feature_names_out(cat_features))
    feature_names = num_features + ohe_features

    # --- Permutation importance ---
    perm = permutation_importance(
        clf, X_test, y_test,
        n_repeats=5,
        random_state=42
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)

st.dataframe(importance_df.head(15))

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Key Drivers of Attrition",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Unable to generate importance chart. Check logs or uploaded file format.")
    st.exception(e)


# ------------------------------------
# Download scored file
# ------------------------------------
st.header("⬇ Download Scored Dataset")

download_df = df.copy()
download_df["AttritionRisk"] = (download_df["AttritionRisk"] * 100).round(2)

csv = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Scored Employee File (CSV)",
    csv,
    "attrition_scored.csv",
    "text/csv"
)






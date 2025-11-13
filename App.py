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


# ---------------------------
# 1. TITLE + SIDEBAR UPLOAD
# ---------------------------
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.title("HR Attrition Risk Dashboard")

st.sidebar.header("Upload HR Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# ---------------------------
# 2. LOAD DATA
# ---------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


if uploaded:
    df = load_data(uploaded)
    st.success("Custom dataset loaded.")
else:
    # Fallback to repository CSV (must be inside GitHub repo)
    df = load_data("hr_attrition_scored.csv")
    st.info("Using default dataset from repository.")


# ---------------------------
# 3. TARGET ENCODING
# ---------------------------
if "Attrition" not in df.columns:
    st.error("Dataset must contain the column 'Attrition'.")
    st.stop()

df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})


# ---------------------------
# 4. PREPARE DATA FOR MODEL
# ---------------------------
drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

X = df.drop(columns=["AttritionFlag", "Attrition"])
y = df["AttritionFlag"]

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = [c for c in X.columns if c not in num_features]

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

base_model = HistGradientBoostingClassifier(random_state=42)
model = Pipeline([
    ("pre", preprocess),
    ("cal", CalibratedClassifierCV(base_model, cv=3))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)


# ---------------------------
# 5. SCORE ENTIRE DATASET
# ---------------------------
df["AttritionRisk"] = model.predict_proba(X)[:, 1]

def bucket(p):
    if p >= 0.40:
        return "High"
    elif p >= 0.20:
        return "Medium"
    return "Low"

df["RiskTier"] = df["AttritionRisk"].apply(bucket)


# ---------------------------
# 6. DEPARTMENT SUMMARY
# ---------------------------
st.subheader("🏢 Department Risk Overview")

dept_summary = (
    df.groupby("Department")
      .agg(
          Employees=("AttritionFlag", "count"),
          Avg_Risk=("AttritionRisk", "mean"),
          High_Risk=("RiskTier", lambda s: (s=="High").sum())
      )
      .reset_index()
)

fig = px.bar(
    dept_summary,
    x="Department",
    y="Avg_Risk",
    color="Avg_Risk",
    color_continuous_scale="Reds",
    title="Average Predicted Attrition Risk by Department"
)
st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# 7. TOP RISK EMPLOYEES
# ---------------------------
st.subheader("Highest-Risk Employees (Top 20)")

if "JobRole" not in df.columns:
    st.error("Dataset missing required column: JobRole")
else:
    top_risk = df.sort_values("AttritionRisk", ascending=False).head(20)

    top_risk["AttritionRisk"] = (top_risk["AttritionRisk"] * 100).round(1)

    st.dataframe(
        top_risk[
            ["Department", "JobRole", "MonthlyIncome", "WorkLifeBalance",
             "AttritionRisk", "RiskTier"]
        ]
    )


# ---------------------------
# 8. FEATURE IMPORTANCE FIXED
# ---------------------------
st.subheader("What Drives Attrition? (Feature Importance)")

# Correctly extract feature names AFTER one-hot encoding
ohe = model.named_steps["pre"].transformers_[1][1]
ohe_features = ohe.get_feature_names_out(cat_features)
feature_names = num_features + list(ohe_features)

# Compute permutation importance
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=5, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False).head(20)

fig2 = px.bar(
    importance_df.sort_values("Importance"),
    x="Importance", y="Feature", orientation="h",
    title="Top Factors Driving Attrition"
)
st.plotly_chart(fig2, use_container_width=True)







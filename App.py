import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# 1. LOAD DATA
# ---------------------------
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

st.title("HR Attrition Risk Dashboard")

uploaded = st.sidebar.file_uploader("Upload scored HR file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # ---------------------------
    # 2. ENSURE REQUIRED COLUMNS
    # ---------------------------
    required_cols = [
        "EmployeeNumber", "Department", "JobRole", "MonthlyIncome",
        "WorkLifeBalance", "AttritionRisk", "RiskTier"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        st.stop()

    # ------------------------------------------
    # 3. AUTO-GENERATE ACTION RECOMMENDATIONS
    # ------------------------------------------
    def generate_action(row):
        if row["RiskTier"] == "High":
            return "Immediate stay interview + supervisor follow-up"
        elif row["RiskTier"] == "Medium":
            return "Monitor monthly + check-in meeting"
        else:
            return "No action needed"

    df["Action"] = df.apply(generate_action, axis=1)

    # ------------------------------------------
    # 4. TOP RISK EMPLOYEES TABLE
    # ------------------------------------------
    st.subheader("Highest-Risk Employees (Top 20)")

    top_risk = df.sort_values("AttritionRisk", ascending=False).head(20)

    display_cols = [
        "EmployeeNumber", "Department", "JobRole",
        "MonthlyIncome", "WorkLifeBalance",
        "AttritionRisk", "RiskTier", "Action"
    ]

    st.dataframe(top_risk[display_cols], height=400)

    # ------------------------------------------
    # 5. DEPARTMENT RISK BAR CHART
    # ------------------------------------------
    st.subheader("Average Risk by Department")

    dept_summary = (
        df.groupby("Department")
        .agg(avg_risk=("AttritionRisk", "mean"))
        .reset_index()
    )
    dept_summary["avg_risk"] = dept_summary["avg_risk"] * 100

    fig = px.bar(
        dept_summary,
        x="Department",
        y="avg_risk",
        color="avg_risk",
        color_continuous_scale="Reds",
        title="Average Predicted Attrition Risk (%) by Department",
        labels={"avg_risk": "Risk (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 6. HEATMAP BY DEPARTMENT & JOB ROLE
    # ------------------------------------------
    st.subheader("Risk Heatmap – Department × Job Role")

    heat_df = (
        df.groupby(["Department", "JobRole"])
        .agg(avg_risk=("AttritionRisk", "mean"))
        .reset_index()
    )
    heat_df["avg_risk"] = heat_df["avg_risk"] * 100

    fig2 = px.density_heatmap(
        heat_df,
        x="Department",
        y="JobRole",
        z="avg_risk",
        color_continuous_scale="Reds",
        title="Risk (%) by Department & Job Role",
        labels={"avg_risk": "Risk (%)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Please upload a scored HR attrition file (CSV) to continue.")

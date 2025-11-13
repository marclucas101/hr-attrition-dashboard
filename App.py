import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="HR Attrition Risk Dashboard",
    layout="wide"
)

# ------------------
# LOAD DATA
# ------------------
@st.cache_data
def load_data():
    return pd.read_csv("hr_attrition_scored.csv")

df = load_data()

st.title("HR Attrition Risk Dashboard")
st.markdown("This dashboard helps HR identify employees at risk of attrition and take early intervention.")

# ------------------
# SUMMARY METRICS
# ------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", len(df))

with col2:
    st.metric("High-Risk Employees", (df["RiskTier"] == "High").sum())

with col3:
    st.metric("Average Predicted Risk (%)",
              round(df["AttritionRisk"].mean() * 100, 2))

# ------------------
# DEPARTMENT RISK VIEW
# ------------------
st.header("Department-Level Risk")

dept = (
    df.groupby("Department")
      .agg(
          Average_Risk=("AttritionRisk", "mean"),
          High_Risk=("RiskTier", lambda s: (s == "High").sum()),
          Employees=("EmployeeNumber", "count")
      )
      .reset_index()
)

dept["Average_Risk"] = dept["Average_Risk"] * 100

fig = px.bar(
    dept,
    x="Department",
    y="Average_Risk",
    color="Average_Risk",
    color_continuous_scale="Reds",
    title="Average Predicted Attrition Risk (%) by Department",
)

st.plotly_chart(fig, use_container_width=True)

# ------------------
# TOP AT-RISK EMPLOYEES
# ------------------
st.header("Top 20 At-Risk Employees")

top20 = (
    df.sort_values("AttritionRisk", ascending=False)
      .head(20)[
          [
            "EmployeeNumber", "Department", "JobRole", "Age", "MonthlyIncome",
            "WorkLifeBalance", "RelationshipSatisfaction",
            "BusinessTravel", "AttritionRisk", "RiskTier"
          ]
      ]
)

top20["AttritionRisk"] = (top20["AttritionRisk"] * 100).round(1)

st.dataframe(top20)

# ------------------
# DOWNLOAD BUTTON
# ------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Full Scored Employee File",
    data=csv,
    file_name="hr_attrition_scored.csv",
    mime="text/csv"
)

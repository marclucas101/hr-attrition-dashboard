# HR Attrition Risk Dashboard (DAO2702 Project)

This Streamlit app is an interactive **HR attrition risk dashboard** built for the DAO2702 module. It allows HR to:

- Upload a scored HR dataset (with predicted attrition risk per employee), or
- Use the bundled sample file `hr_attrition_scored.csv`
- See **department-level risk** and summary metrics
- Identify **high-risk employees** for early intervention
- Understand **what drives attrition** using:
  - Permutation feature importance
  - SHAP-based global explanations (mean |SHAP|)

---

## 1. Repository Structure

Typical layout:

```text
.
├─ app.py                      # Streamlit app (this file)
├─ hr_attrition_scored.csv     # Sample scored HR dataset
├─ WA_Fn-UseC_-HR-Employee-Attrition.csv   # (optional) raw IBM attrition dataset
├─ requirements.txt            # Python dependencies
└─ README.md                   # Documentation


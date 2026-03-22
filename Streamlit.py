import streamlit as st
import requests

st.set_page_config(page_title="CreditGuard", page_icon="💳", layout="wide")

# -------------------------------
# 🎨 STYLING
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}
.stWidget label {
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.85rem !important;
    color: #475569 !important;
    font-weight: 600 !important;
}
[data-testid="column"] {
    padding-left: 5px !important;
    padding-right: 5px !important;
}
.category-header {
    color: #1E3A8A;
    font-weight: 700;
    font-size: 1.3rem;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 5px;
}
.stButton>button {
    margin-top: 20px;
    width: 100%;
    background-color: #1E3A8A;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 CreditGuard - Credit Risk Prediction")

# -------------------------------
# 📋 FORM
# -------------------------------
with st.form("credit_form", border=False):

    # PERSONAL
    st.markdown('<div class="category-header">👤 Personal Information</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1: age = st.number_input("Age", 18, 100, 25)
    with p2: years_at_current_address = st.number_input("Years at Address", 0, 50, 2)
    with p3: number_of_dependants = st.number_input("Dependants", 0, 10, 0)
    with p4: income = st.number_input("Income ($)", 0, 1000000, 1000)
    with p5: residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Other"])
    with p6: bank_balance_at_application = st.number_input("Bank Balance", 0, 1000000, 500)

    # LOAN
    st.markdown('<div class="category-header">💰 Loan Details</div>', unsafe_allow_html=True)
    l1, l2, l3, l4, l5 = st.columns(5)

    with l1: loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
    with l2: loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Home", "Personal", "Education"])
    with l3: loan_amount = st.number_input("Loan Amount", 0, 1000000, 1000)
    with l4: loan_tenure_months = st.number_input("Loan Tenure Months", 1, 360, 12)
    with l5: total_loan_months = st.number_input("Total Loan Months", 0, 1000, 12)

    # CREDIT
    st.markdown('<div class="category-header">📊 Credit History</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1: number_of_open_accounts = st.number_input("Open Accounts", 0, 50, 0)
    with c2: number_of_closed_accounts = st.number_input("Closed Accounts", 0, 50, 0)
    with c3: credit_utilization_ratio = st.number_input("Credit Utilization (0-1)", 0.0, 1.0, 0.0)
    with c4: delinquent_months = st.number_input("Delinquent Months", 0, 50, 0)
    with c5: total_dpd = st.number_input("Total DPD", 0, 1000, 0)
    with c6: enquiry_count = st.number_input("Enquiry Count", 0, 100, 0)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        submitted = st.form_submit_button("Predict")

# -------------------------------
# 🚀 PREDICTION
# -------------------------------
# ------------------------------- 
# 🚀 PREDICTION & DISPLAY
# ------------------------------- 
if submitted:
    payload = {
        "age": age,
        "number_of_dependants": number_of_dependants,
        "years_at_current_address": years_at_current_address,
        "income": income,
        "bank_balance_at_application": bank_balance_at_application,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "total_loan_months": total_loan_months,
        "number_of_open_accounts": number_of_open_accounts,
        "number_of_closed_accounts": number_of_closed_accounts,
        "enquiry_count": enquiry_count,
        "credit_utilization_ratio": credit_utilization_ratio,
        "delinquent_months": delinquent_months,
        "total_dpd": total_dpd,
        "residence_type": residence_type
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()

        st.markdown("---")

        # -------------------------------
        # ✅ Decision
        # -------------------------------
        decision = result.get("decision", "N/A")
        risk_level = result.get("risk_level", "N/A")
        message = result.get("message", "")

        if decision.upper() == "APPROVE":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # -------------------------------
        # 📌 Summary
        # -------------------------------
        st.markdown(f"""
        ### 📌 Summary

        - **Decision:** {decision}  
        - **Risk Level:** {risk_level}  
        - **Confidence:** {round(result.get("probability", 0), 2)}%  
        """)

        # -------------------------------
        # 💳 CREDIT SCORE (NEW)
        # -------------------------------
        credit_score = result.get("credit_score", 0)
        score_band = result.get("score_band", "N/A")

        st.markdown("## 💳 Credit Score")

        if credit_score >= 750:
            score_color = "#16a34a"
        elif credit_score >= 650:
            score_color = "#84cc16"
        elif credit_score >= 550:
            score_color = "#f59e0b"
        else:
            score_color = "#dc2626"

        st.markdown(f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:12px;
            text-align:center;
            box-shadow:0px 4px 10px rgba(0,0,0,0.05);
        ">
            <div style="font-size:18px; color:#475569;">Credit Score</div>
            <div style="font-size:48px; font-weight:800; color:{score_color};">
                {credit_score}
            </div>
            <div style="font-size:16px; font-weight:600; color:#1E3A8A;">
                {score_band}
            </div>
        </div>
        """, unsafe_allow_html=True)

        score_percent = (credit_score - 300) / 600 * 100

        st.markdown(f"""
        <div style="margin-top:15px;">
            <div style="
                background-color:#e5e7eb;
                border-radius:10px;
                height:18px;
                width:100%;
            ">
                <div style="
                    background-color:{score_color};
                    width:{score_percent}%;
                    height:100%;
                    border-radius:10px;
                    text-align:right;
                    padding-right:5px;
                    color:white;
                    font-size:12px;
                    font-weight:600;
                ">
                    {credit_score}
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.caption("Score ranges: 750+ Excellent | 650–749 Good | 550–649 Fair | Below 550 Poor")

        # -------------------------------
        # 📊 Metrics
        # -------------------------------
        default_prob = result.get("probability", 0) / 100
        approval_prob = 1 - default_prob

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Default Risk", f"{round(default_prob*100,2)}%")
        with col2:
            st.metric("🟢 Approval Chance", f"{round(approval_prob*100,2)}%")

        if message:
            st.info(message)

        # -------------------------------
        # 📊 Approval Probability Bar
        # -------------------------------
        st.subheader("📊 Approval Probability")

        if approval_prob > 0.7:
            bar_color = "#16a34a"
            label = "High Approval Chance"
        elif approval_prob > 0.4:
            bar_color = "#f59e0b"
            label = "Moderate Approval Chance"
        else:
            bar_color = "#dc2626"
            label = "Low Approval Chance"

        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <b>{label}</b> ({round(approval_prob*100,2)}%)
        </div>

        <div style="
            background-color:#e5e7eb;
            border-radius:10px;
            height:18px;
            width:100%;
        ">
            <div style="
                background-color:{bar_color};
                width:{approval_prob*100}%;
                height:100%;
                border-radius:10px;
                text-align:right;
                padding-right:5px;
                color:white;
                font-size:12px;
                font-weight:600;
            ">
                {round(approval_prob*100,1)}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------
        # ⚡ Top Factors
        # -------------------------------
        top_factors = result.get("top_factors", [])

        if top_factors:
            st.subheader("⚡ Key Factors Influencing Decision")

            for factor in top_factors:
                if "Increased Risk" in factor:
                    st.markdown(f"""
                    <div style="background-color:#fee2e2; padding:12px; border-radius:8px; margin-bottom:8px; border-left:5px solid #dc2626;">
                        🔴 <b>{factor}</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#dcfce7; padding:12px; border-radius:8px; margin-bottom:8px; border-left:5px solid #16a34a;">
                        🟢 <b>{factor}</b>
                    </div>
                    """, unsafe_allow_html=True)

        # -------------------------------
        # 🧠 Feature Summary
        # -------------------------------
        feature_summary = result.get("feature_summary", {})

        if feature_summary:
            st.subheader("🧠 Feature Insights")

            for feat, info in feature_summary.items():
                st.markdown(f"""
                <div style="
                    background-color:#ffffff;
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
                ">
                    <b style="color:#1E3A8A;">{feat.replace("_", " ").title()}</b><br>
                    <span><b>Value:</b> {info['value']}</span><br>
                    <span style="color:#475569;">{info['explanation']}</span>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
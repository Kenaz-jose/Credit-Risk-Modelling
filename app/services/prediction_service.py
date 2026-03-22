import pandas as pd
from app.schemas.application import LoanApplicationRequest,LoanApplicationResponse
import shap

# Feature explanations for frontend
FEATURE_EXPLANATIONS = {
    "loan_to_income": "Ratio of the requested loan amount to the user's annual income. Higher values indicate more financial burden.",
    "delinquent_ratio": "Proportion of months where the user was delinquent on previous loans.",
    "avg_dpd_per_delinquency": "Average number of days past due per delinquent loan.",
    "age": "Age of the applicant in years.",
    "number_of_dependants": "Number of people financially dependent on the applicant.",
    "years_at_current_address": "Number of years the applicant has lived at their current address.",
    "loan_tenure_months": "Number of months the loan will last.",
    "bank_balance_at_application": "Current bank balance of the applicant at the time of application.",
    "number_of_open_accounts": "Number of open credit accounts the applicant currently has.",
    "number_of_closed_accounts": "Number of closed credit accounts in the past.",
    "enquiry_count": "Number of credit inquiries made recently.",
    "credit_utilization_ratio": "Portion of available credit currently being used.",
    "residence_type": "Type of residence: Owned, Rented, or Mortgage.",
    "loan_purpose": "Purpose of the loan: Auto, Education, Home, or Personal.",
    "loan_type": "Secured or Unsecured loan type.",
}
def make_prediction(request: LoanApplicationRequest, model):
    # 1️⃣ Compute features
    features_dict = request.compute_features()
    X = pd.DataFrame([features_dict])

    # 2️⃣ Predict probability
    probability = model.predict_proba(X)[0][1]
    default_risk = round(probability * 100, 2)

    # -------------------------------
    # 🆕 3️⃣ Credit Score
    # -------------------------------
    credit_score = int(300 + (1 - probability) * 600)

    if credit_score >= 750:
        score_band = "EXCELLENT"
    elif credit_score >= 650:
        score_band = "GOOD"
    elif credit_score >= 550:
        score_band = "FAIR"
    else:
        score_band = "POOR"

    # -------------------------------
    # 4️⃣ Risk level & decision
    # -------------------------------
    if probability >= 0.8:
        risk_level = "HIGH"
        decision = "REJECT"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
        decision = "REVIEW"
    else:
        risk_level = "LOW"
        decision = "APPROVE"

    message = f"Customer is {risk_level.lower()} risk and decision is {decision}."

    # -------------------------------
    # 5️⃣ Model coefficients
    # -------------------------------
    clf = model.model.estimator
    preprocessor = model.preprocessor
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    # -------------------------------
    # 6️⃣ Contributions
    # -------------------------------
    contributions = {}
    X_transformed = preprocessor.transform(X)[0]

    for feat, coef, val in zip(feature_names, coefs, X_transformed):
        contributions[feat] = coef * val

    sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    top_factors = []
    feature_summary = {}

    for feat, contrib in sorted_feats:
        raw_key = feat.replace("num__", "").replace("cat__", "")

        if "_" in raw_key and raw_key.split("_")[0] in FEATURE_EXPLANATIONS:
            base_key = raw_key.split("_")[0]
            explanation = FEATURE_EXPLANATIONS.get(base_key, "")
            value = features_dict.get(base_key)
        else:
            value = features_dict.get(raw_key)
            explanation = FEATURE_EXPLANATIONS.get(raw_key, "")

        label = "Increased Risk" if contrib > 0 else "Reduced Risk"
        top_factors.append(f"{raw_key.replace('_',' ').title()} {label}")

        feature_summary[raw_key] = {
            "value": value,
            "explanation": explanation
        }

    # -------------------------------
    # 7️⃣ Response
    # -------------------------------
    response = LoanApplicationResponse(
        probability=default_risk,
        credit_score=credit_score,      # ✅ added
        score_band=score_band,          # ✅ added
        risk_level=risk_level,
        decision=decision,
        message=message,
        top_factors=top_factors,
        top_feature_keys=[feat for feat, _ in sorted_feats],
        feature_summary=feature_summary
    )

    return response
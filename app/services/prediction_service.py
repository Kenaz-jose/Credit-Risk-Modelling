import logging
import pandas as pd
import numpy as np

from app.schemas.application import (
    LoanApplicationRequest,
    LoanApplicationResponse,
    FeatureInsight
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_EXPLANATIONS = {
    "loan_to_income": "Ratio of the requested loan amountto the user's annual income.",

    "delinquent_ratio": "Proportion of months where the user was delinquent on previous loans.",

    "avg_dpd_per_delinquency": "Average number of days past due per delinquent loan.",

    "age": "Age of the applicant in years.",

    "number_of_dependants":  "Number of financially dependent people.",

    "years_at_current_address": "Years at current residence.",

    "loan_tenure_months": "Loan duration in months.",

    "bank_balance_at_application": "Bank balance at application time.",

    "number_of_open_accounts": "Currently open credit accounts.",

    "number_of_closed_accounts": "Previously closed credit accounts.",

    "enquiry_count": "Recent credit enquiries.",

    "credit_utilization_ratio": "Portion of available credit used.",

    "residence_type": "Owned, rented, or mortgage residence.",

    "loan_purpose": "Purpose of the loan.",

    "loan_type": "Secured or unsecured loan.",
}


def validate_and_clean_features(features_dict):

    if features_dict["income"] <= 0:
       raise ValueError("Income must be greater than zero")

    utilization = features_dict["credit_utilization_ratio"]

    if utilization > 1:
        utilization = utilization / 100

    utilization = max(0, min(utilization, 1))

    features_dict["credit_utilization_ratio"] = utilization

    loan_to_income = (features_dict["loan_amount"]/ max(features_dict["income"], 1))

    # Clip extreme values
    loan_to_income = min(loan_to_income, 20)

    features_dict["loan_to_income"] = loan_to_income

    delinquent_ratio = (features_dict["delinquent_months"]/ max(features_dict["total_loan_months"],1))

    delinquent_ratio = min(delinquent_ratio, 1)

    features_dict["delinquent_ratio"] = delinquent_ratio

    avg_dpd = (features_dict["total_dpd"]/ max(features_dict["delinquent_months"],1))

    avg_dpd = min(avg_dpd, 365)

    features_dict["avg_dpd_per_delinquency"] = avg_dpd

    return features_dict

def make_prediction(
    request: LoanApplicationRequest,
    model
):

    logger.info("========== PREDICTION STARTED ==========")

    features_dict = request.model_dump()

    logger.info(f"Raw Request Features: {features_dict}")

    features_dict = validate_and_clean_features(features_dict)

    logger.info(f"Validated Features: {features_dict}")

    X = pd.DataFrame([features_dict])

    probabilities = model.predict_proba(X)

    logger.info(f"Predicted Probabilities: {probabilities}")

    probability = float(probabilities[0][1])

    probability = float(np.clip(probability, 0.01, 0.99))
    
    logger.info(f"Default Probability: {probability}")

    default_risk = round(probability * 100, 2)

    credit_score = int(300 + (1 - probability) * 600)

    credit_score = max(300,min(credit_score, 900))

    if credit_score >= 750:
        score_band = "EXCELLENT"

    elif credit_score >= 650:
        score_band = "GOOD"

    elif credit_score >= 550:
        score_band = "FAIR"

    else:
        score_band = "POOR"

    manual_review = False

    if features_dict["enquiry_count"] >=20:
        manual_review = True
    if features_dict["credit_utilization_ratio"] > 0.9:
        manual_review = True
    if features_dict["delinquent_ratio"] > 0.5:
        manual_review = True


    if probability > 0.5:

        risk_level = "HIGH"
        decision = "REJECT"

    elif 0.3 <= probability <= 0.5:

        risk_level = "MEDIUM"
        decision = "REVIEW"

    else:

        risk_level = "LOW"
        decision = "APPROVE"
    
    if manual_review and decision == "APPROVE":
        decision = "REVIEW"

    message = (
        f"Customer is {risk_level.lower()} "
        f"risk and decision is {decision}."
    )

    clf = model.model.estimator

    logger.info(f"Model Classes: {clf.classes_}")

    preprocessor = model.preprocessor

    feature_names = (preprocessor.get_feature_names_out())

    coefs = clf.coef_[0]

    X_transformed = preprocessor.transform(X)[0]

    logger.info("========== TRANSFORMED FEATURES ==========")

    for feat, val in zip(feature_names,X_transformed):
        logger.info(f"{feat}: {val}")

    contributions = {}

    logger.info("========== FEATURE CONTRIBUTIONS ==========")

    for feat, coef, val in zip(feature_names,coefs,X_transformed):

        contrib = float(coef * val)

        logger.info(
            f"Feature: {feat} | "
            f"Value: {val:.4f} | "
            f"Coef: {coef:.4f} | "
            f"Contribution: {contrib:.4f}"
        )

        contributions[feat] = contrib

    sorted_feats = sorted(contributions.items(),key=lambda x: abs(x[1]),reverse=True)[:5]

    top_factors = []

    top_feature_insights = []

    feature_summary = {}

    logger.info("========== TOP FEATURES ==========")

    for feat, contrib in sorted_feats:

        logger.info(
            f"Top Feature: {feat} | "
            f"Contribution: {contrib:.4f}"
        )

        raw_key = (
            feat.replace("num__", "")
            .replace("cat__", "")
        )

        # Better mapping for one-hot encoded features
        matched_key = next(
            (
                k
                for k in FEATURE_EXPLANATIONS.keys()
                if raw_key.startswith(k)
            ),
            raw_key
        )

        value = features_dict.get(matched_key)

        explanation = FEATURE_EXPLANATIONS.get(matched_key,"")

        label = (
            "Increased Risk"
            if contrib > 0
           else "Reduced Risk")

        top_factors.append(
            f"{raw_key.replace('_', ' ').title()} "
            f"{label}"
        )
        
        top_feature_insights.append(
            FeatureInsight(
                name=raw_key.replace("_", " ").title(),
                effect=label,
                explanation=explanation
            )
        )
        feature_summary[raw_key] = {
            "value": value,
            "explanation": explanation,
            "contribution": round(contrib, 4)
        }

    response = LoanApplicationResponse(
        probability=default_risk,
        credit_score=credit_score,
        score_band=score_band,
        risk_level=risk_level,
        decision=decision,
        message=message,
        top_factors=top_factors,
        top_feature_insights=top_feature_insights,
        top_feature_keys=[
            feat for feat, _ in sorted_feats
        ],
        feature_summary=feature_summary
    )

    logger.info("========== PREDICTION COMPLETED ==========")

    return response

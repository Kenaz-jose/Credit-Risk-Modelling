from pydantic import BaseModel, Field, model_validator
from typing import Literal,List,Dict,Any
from typing import List, Dict, Any, Literal

class LoanApplicationRequest(BaseModel):
    #  Personal Info
    age: int = Field(..., ge=18, le=100)
    number_of_dependants: int = Field(..., ge=0, le=10)
    years_at_current_address: int = Field(..., ge=0, le=80)

    #  Financial Info
    income: float = Field(..., gt=0, le=1e9)
    bank_balance_at_application: float = Field(..., ge=0)

    #  Loan Info
    loan_amount: float = Field(..., gt=0, le=1e9)
    loan_tenure_months: int = Field(..., gt=0,le=600)
    loan_type: Literal["Secured", "Unsecured"]
    loan_purpose: Literal["Auto", "Education", "Home", "Personal"]

    #  Credit Behavior
    number_of_open_accounts: int = Field(..., ge=0)
    number_of_closed_accounts: int = Field(..., ge=0)
    enquiry_count: int = Field(..., ge=0,le=50)
    credit_utilization_ratio: float = Field(..., ge=0, le=1,description="0 to 1 ratio")

    #  Delinquency Info
    delinquent_months: int = Field(..., ge=0)
    total_loan_months: int = Field(..., gt=0)
    total_dpd: float = Field(..., ge=0)

    #  Categorical
    residence_type: Literal["Owned", "Rented", "Mortgage"]

    #  CROSS-FIELD VALIDATION
    @model_validator(mode="after")
    def validate_logic(self):

        total_accounts = self.number_of_open_accounts + self.number_of_closed_accounts
        if total_accounts == 0:
            raise ValueError("User must have at least one account")

        if self.delinquent_months > self.total_loan_months:
            raise ValueError("Delinquent months cannot exceed total loan months")

        if self.delinquent_months == 0 and self.total_dpd > 0:
            raise ValueError("DPD cannot exist if no delinquency")

        if self.years_at_current_address > self.age:
            raise ValueError("Years at current address cannot exceed age")

        if self.years_at_current_address > (self.age - 18):
            raise ValueError("Years at current address unrealistic for given age")

        if self.number_of_dependants > (self.age // 2):
            raise ValueError("Too many dependants for given age")

        return self
    #  COMPUTED FEATURES
    def compute_features(self):

        loan_to_income = self.loan_amount / self.income

        delinquent_ratio = (
            self.delinquent_months / self.total_loan_months
            if self.total_loan_months > 0 else 0
        )

        avg_dpd_per_delinquency = (
            self.total_dpd / self.delinquent_months
            if self.delinquent_months > 0 else 0
        )

        return {
            "age": self.age,
            "number_of_dependants": self.number_of_dependants,
            "years_at_current_address": self.years_at_current_address,
            "loan_tenure_months": self.loan_tenure_months,
            "bank_balance_at_application": self.bank_balance_at_application,
            "number_of_open_accounts": self.number_of_open_accounts,
            "number_of_closed_accounts": self.number_of_closed_accounts,
            "enquiry_count": self.enquiry_count,
            "credit_utilization_ratio": self.credit_utilization_ratio,
            "loan_to_income": loan_to_income,
            "delinquent_ratio": delinquent_ratio,
            "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
            "residence_type": self.residence_type,
            "loan_purpose": self.loan_purpose,
            "loan_type": self.loan_type,
        }
class FeatureInsight(BaseModel):
    name: str          # Human-readable feature name
    effect: str        # "Increased Risk" or "Reduced Risk"
    explanation: str   # Short explanation for the feature


class LoanApplicationResponse(BaseModel):
    probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    credit_score:int
    score_band:str
    message: str
    top_factors: List[str] 
    top_feature_insights: List[FeatureInsight] = []
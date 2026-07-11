from pydantic import BaseModel, Field, model_validator
from typing import Literal, List


# =========================================================
# REQUEST SCHEMA
# =========================================================
class LoanApplicationRequest(BaseModel):

    # -----------------------------------------------------
    # PERSONAL INFO
    # -----------------------------------------------------
    age: int = Field(..., ge=18, le=100)

    number_of_dependants: int = Field(
        ...,
        ge=0,
        le=10
    )

    years_at_current_address: int = Field(
        ...,
        ge=0,
        le=80
    )

    # FINANCIAL INFO
    income: float = Field(
        ...,
        gt=0,
        le=1_000_000_000,
        description="Annual income"
    )

    bank_balance_at_application: float = Field(
        ...,
        ge=0,
        le=1_000_000_000
    )

    # -----------------------------------------------------
    # LOAN INFO
    # -----------------------------------------------------
    loan_amount: float = Field(
        ...,
        gt=0,
        le=1_000_000_000
    )

    loan_tenure_months: int = Field(
        ...,
        gt=0,
        le=600
    )

    loan_type: Literal["Secured", "Unsecured"]

    loan_purpose: Literal[
        "Auto",
        "Education",
        "Home",
        "Personal"
    ]

    # -----------------------------------------------------
    # CREDIT BEHAVIOR
    # -----------------------------------------------------
    number_of_open_accounts: int = Field(..., ge=0)

    number_of_closed_accounts: int = Field(..., ge=0)

    enquiry_count: int = Field(
        ...,
        ge=0,
        le=50
    )

    credit_utilization_ratio: float = Field(
        ...,
        ge=0,
        le=100,
        description="Credit utilization ratio between 0 and 100"
    )

    # -----------------------------------------------------
    # DELINQUENCY INFO
    # -----------------------------------------------------
    delinquent_months: int = Field(..., ge=0)

    total_loan_months: int = Field(..., gt=0)

    total_dpd: float = Field(
        ...,
        ge=0,
        le=3650
    )

    # -----------------------------------------------------
    # RESIDENCE
    # -----------------------------------------------------
    residence_type: Literal[
        "Owned",
        "Rented",
        "Mortgage"
    ]

    # =====================================================
    # CROSS-FIELD VALIDATION
    # =====================================================
    @model_validator(mode="after")
    def validate_logic(self):

        # ---------------------------------------------
        # Minimum account history
        # ---------------------------------------------
        total_accounts = (
            self.number_of_open_accounts
            + self.number_of_closed_accounts
        )

        if total_accounts == 0:
            raise ValueError(
                "Applicant must have at least one account"
            )

        # ---------------------------------------------
        # Delinquency checks
        # ---------------------------------------------
        if self.delinquent_months > self.total_loan_months:
            raise ValueError(
                "Delinquent months cannot exceed total loan months"
            )

        if self.delinquent_months == 0 and self.total_dpd > 0:
            raise ValueError(
                "DPD cannot exist when delinquent months is 0"
            )

        # ---------------------------------------------
        # Address realism
        # ---------------------------------------------
        if self.years_at_current_address > self.age:
            raise ValueError(
                "Years at current address cannot exceed age"
            )

        if self.years_at_current_address > (self.age - 18):
            raise ValueError(
                "Years at current address unrealistic for given age"
            )

        # ---------------------------------------------
        # Dependants realism
        # ---------------------------------------------
        if self.number_of_dependants > (self.age // 2):
            raise ValueError(
                "Too many dependants for given age"
            )

        # ---------------------------------------------
        # Financial sanity checks
        # ---------------------------------------------
        if self.loan_amount > (self.income * 20):
            raise ValueError(
                "Loan amount unrealistically high compared to income"
            )

        if (
            self.bank_balance_at_application
            > (self.income * 50)
        ):
            raise ValueError(
                "Bank balance unrealistic compared to income"
            )

        return self

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    def compute_features(self):

        # ---------------------------------------------
        # Loan-to-income ratio
        # Clipped for model stability
        # ---------------------------------------------
        loan_to_income = min(
            self.loan_amount / self.income,
            10
        )

        # Delinquent ratio
        delinquent_ratio = (
            self.delinquent_months / self.total_loan_months
            if self.total_loan_months > 0
            else 0
        )

        # Avg DPD
        avg_dpd_per_delinquency = (
            self.total_dpd / self.delinquent_months
            if self.delinquent_months > 0
            else 0
        )

        # Final model features
        return {
            "age": self.age,
            "number_of_dependants": self.number_of_dependants,
            "years_at_current_address": self.years_at_current_address,
            "loan_tenure_months": self.loan_tenure_months,
            "bank_balance_at_application": self.bank_balance_at_application,
            "number_of_open_accounts": self.number_of_open_accounts,
            "number_of_closed_accounts": self.number_of_closed_accounts,
            "enquiry_count": self.enquiry_count,

            # Already validated 0-1
            "credit_utilization_ratio": min(
                self.credit_utilization_ratio,
                100
            ),

            "loan_to_income": loan_to_income,
            "delinquent_ratio": delinquent_ratio,
            "avg_dpd_per_delinquency": avg_dpd_per_delinquency,

            "residence_type": self.residence_type,
            "loan_purpose": self.loan_purpose,
            "loan_type": self.loan_type,
        }


# FEATURE INSIGHT RESPONSE
class FeatureInsight(BaseModel):
    name: str
    effect: str
    explanation: str


# RESPONSE SCHEMA
class LoanApplicationResponse(BaseModel):

    probability: float

    risk_level: Literal[
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    decision: Literal[
        "APPROVE",
        "REVIEW",
        "REJECT"
    ]

    credit_score: int

    score_band: str

    message: str

    top_factors: List[str]

    top_feature_insights: List[FeatureInsight] = []

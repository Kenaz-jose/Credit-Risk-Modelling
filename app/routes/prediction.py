from fastapi import APIRouter, HTTPException
from app.schemas.application import LoanApplicationRequest, LoanApplicationResponse
from app.services.prediction_service import make_prediction
import pickle

router = APIRouter(tags=["Prediction"])

with open("final_model/model.pkl", "rb") as f:
    model = pickle.load(f)

@router.post(
    "/predict",
    response_model=LoanApplicationResponse,
    status_code=200,
    summary="Predict loan default risk",
    description="Returns prediction (0/1) and probability of default."
)
def predict_application(request: LoanApplicationRequest):
    # request is now a LoanApplicationRequest object, not dict
    response = make_prediction(request, model)
    return response
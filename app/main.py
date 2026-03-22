# app/main.py
from fastapi import FastAPI
from app.routes import prediction

app = FastAPI(title="Credit Risk Prediction API")

# Include router
app.include_router(prediction.router)
"""
GraphShield AI API
"""

from fastapi import FastAPI

from api.schemas import Transaction
from api.predict import predict_transaction

app = FastAPI(
    title="GraphShield AI",
    version="1.0.0",
    description="Fraud Detection using Machine Learning and Graph Neural Networks"
)


@app.get("/")
def home():

    return {
        "message": "GraphShield AI API Running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    result = predict_transaction(
        transaction.model_dump()
    )

    return result
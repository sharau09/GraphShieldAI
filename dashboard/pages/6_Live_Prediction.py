import streamlit as st
import requests

st.set_page_config(
    page_title="Live Prediction",
    layout="wide"
)

st.title("🔍 Live Fraud Prediction")

st.write("Enter transaction details below and click Predict.")

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ]
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0,
    value=5000.0
)

newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0,
    value=4000.0
)

oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0,
    value=1000.0
)

if st.button("Predict Fraud"):

    payload = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Completed")

            st.json(result)

        else:

            st.error(
                f"API Error : {response.status_code}"
            )

    except Exception as e:

        st.error(str(e))
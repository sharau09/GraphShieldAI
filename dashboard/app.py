import streamlit as st

st.set_page_config(
    page_title="GraphShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ GraphShield AI")

st.subheader(
    "AI Powered Fraud Detection using Machine Learning and Graph Neural Networks"
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Models", "4")
col2.metric("API", "Running")
col3.metric("Dataset", "PaySim")

st.markdown("---")

st.success("✔ Random Forest")
st.success("✔ XGBoost")
st.success("✔ GCN")
st.success("✔ GraphSAGE")

st.info("Use the left sidebar to navigate.")
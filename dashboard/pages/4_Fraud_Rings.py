import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🕸 Fraud Rings")

csv = Path("outputs/reports/fraud_rings.csv")

if not csv.exists():
    st.warning("Run main.py first.")
    st.stop()

df = pd.read_csv(csv)

st.metric(
    "Suspicious Communities",
    len(df)
)

st.dataframe(
    df,
    use_container_width=True
)
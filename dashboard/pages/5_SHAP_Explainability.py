import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="SHAP Explainability",
    layout="wide"
)

st.title("🧠 SHAP Explainability")

summary = Path("outputs/explainability/shap_summary.png")
bar = Path("outputs/explainability/shap_bar.png")
csv = Path("outputs/explainability/feature_importance.csv")

if summary.exists():

    st.subheader("SHAP Summary Plot")

    st.image(summary, use_container_width=True)

else:

    st.warning("shap_summary.png not found.")

st.divider()

if bar.exists():

    st.subheader("Feature Importance")

    st.image(bar, use_container_width=True)

st.divider()

if csv.exists():

    df = pd.read_csv(csv)

    st.subheader("Feature Ranking")

    st.dataframe(
        df,
        use_container_width=True
    )

else:

    st.warning("feature_importance.csv not found.")
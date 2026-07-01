import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.title("📊 Model Comparison")

df = pd.read_csv("outputs/reports/model_comparison.csv")

st.dataframe(df, use_container_width=True)

st.markdown("---")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

metric = st.selectbox(
    "Select Metric",
    metrics
)

fig = px.bar(
    df,
    x="Model",
    y=metric,
    color="Model",
    text=metric,
    title=f"{metric} Comparison"
)

fig.update_traces(textposition="outside")

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.subheader("Performance Summary")

best = df.sort_values(
    metric,
    ascending=False
).iloc[0]

st.success(
    f"🏆 Best {metric}: {best['Model']} ({best[metric]:.4f})"
)
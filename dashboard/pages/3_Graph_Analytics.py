import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Graph Analytics",
    layout="wide"
)

st.title("📈 Graph Analytics")

# ======================================================
# Metrics
# ======================================================

col1, col2, col3 = st.columns(3)

col1.metric("Nodes", "192,912")
col2.metric("Edges", "100,000")
col3.metric("Density", "0.00000269")

st.divider()

# ======================================================
# Network Visualization
# ======================================================

st.subheader("Customer Transaction Network")

image_path = Path("outputs/graphs/customer_network.png")

if image_path.exists():
    image = Image.open(image_path)
    st.image(image, use_container_width=True)
else:
    st.warning("customer_network.png not found.")

st.divider()

# ======================================================
# Graph Features
# ======================================================

csv_path = Path("outputs/reports/graph_features.csv")

if csv_path.exists():

    df = pd.read_csv(csv_path)

    st.subheader("Top 20 Customers by Degree")

    top = (
        df.sort_values(
            "degree",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(top, use_container_width=True)

    st.divider()

    fig = px.bar(
        top,
        x="customer",
        y="degree",
        color="degree",
        title="Top Customers by Degree"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("PageRank Distribution")

    fig = px.histogram(
        df,
        x="pagerank",
        nbins=40,
        title="PageRank Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.error("graph_features.csv not found.")
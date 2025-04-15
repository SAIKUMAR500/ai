import streamlit as st
import pandas as pd
from app.chatbot import get_response
from app.intent_classifier import classify_intent
from app.models.linear_regression import run_linear_regression
from app.models.kmeans import run_kmeans_clustering
from app.models.autoencoder import run_autoencoder
from app.utils import preprocess_data

st.set_page_config(page_title="ML Chatbot", layout="wide")
st.title("🤖 Chatbot with Machine Learning Capabilities")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
data = None

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
        st.write("### Dataset Preview")
        st.dataframe(data.head())
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

st.write("---")
user_input = st.text_input("Ask me something:", placeholder="E.g., predict house price, cluster customers, compress data")

if user_input:
    intent = classify_intent(user_input)
    st.write(f"🧠 Detected Intent: **{intent}**")

    if intent == "linear_regression":
        if data is not None:
            run_linear_regression(data)
        else:
            st.warning("Please upload a dataset to use this model.")

    elif intent == "kmeans":
        if data is not None:
            run_kmeans_clustering(data)
        else:
            st.warning("Please upload a dataset to use this model.")

    elif intent == "autoencoder":
        if data is not None:
            run_autoencoder(data)
        else:
            st.warning("Please upload a dataset to use this model.")

    else:
        response = get_response(user_input)
        st.write(response)
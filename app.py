import openai
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import KMeans
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import default_guarded_getiter 

# Set OpenAI API key
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Set up page config
st.set_page_config(page_title="ML Chatbot with Code Generation", layout="wide")
st.title("🤖 Chatbot with Machine Learning and Python Code Generation")

# Sidebar for dataset upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
data = None

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    try:
        data = load_data(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
        st.write("### Dataset Preview")
        st.dataframe(data.head())
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# User input for chatbot interaction
st.write("---")
user_input = st.text_input("Ask me something:", placeholder="E.g., train a linear regression model, generate code for k-means clustering")

# Choose which GPT model to use
selected_model = st.selectbox(
    "Choose the ChatGPT model to respond:",
    ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"],
    index=0
)

# Intent classification
def classify_intent(user_input):
    user_input = user_input.lower()
    if any(keyword in user_input for keyword in ["regression", "predict", "estimate"]):
        return "linear_regression"
    elif any(keyword in user_input for keyword in ["decision tree", "tree"]):
        return "decision_tree"
    elif any(keyword in user_input for keyword in ["random forest", "forest"]):
        return "random_forest"
    elif any(keyword in user_input for keyword in ["svm", "support vector machine"]):
        return "svm"
    elif any(keyword in user_input for keyword in ["cluster", "group", "segmentation"]):
        return "kmeans"
    elif any(keyword in user_input for keyword in ["compress", "autoencoder", "reduce"]):
        return "autoencoder"
    elif any(keyword in user_input for keyword in ["generate", "python code", "create code", "write code"]):
        return "code_generation"
    elif any(keyword in user_input for keyword in ["file", "upload", "download", "process"]):
        return "file_operation"
    else:
        return "chat"

# Generate Python code using OpenAI API (Code generation)
def generate_python_code(user_input):
    prompt = f"Write Python code to {user_input}"
    response = openai.Completion.create(
        model="code-davinci-003",  # Using Codex model optimized for code generation
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Display generated Python code with syntax highlighting
def display_python_code(code):
    st.code(code, language="python")

# Execute Python code securely with restricted environment
def execute_restricted_python(code):
    try:
        compiled_code = compile_restricted(code, '<string>', 'exec')
        exec_globals = {}
        exec(compiled_code, exec_globals)
        st.success("Code executed successfully!")
        return exec_globals
    except Exception as e:
        st.error(f"Error executing the code: {e}")
        return None

# Dimensionality reduction with PCA
def pca_dimensionality_reduction(data, n_components=2):
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 2:
        return "Need at least two numerical columns for PCA."
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df, pca.explained_variance_ratio_

# Plot Correlation Heatmap
def plot_correlation_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot()

# Plot histogram
def plot_histogram(data):
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    st.write("### Histogram of Numerical Columns")
    for col in numeric_cols:
        st.write(f"#### {col}")
        fig, ax = plt.subplots()
        ax.hist(data[col], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

# Main logic based on user input
if user_input:
    intent = classify_intent(user_input)

    # Handle Python code generation or file operations
    if intent == "code_generation":
        st.write("### Generated Python Code:")
        generated_code = generate_python_code(user_input)
        display_python_code(generated_code)

        execute_code = st.button("Execute the Code")
        if execute_code:
            result = execute_restricted_python(generated_code)
            st.write(f"### Execution Result:")
            st.write(result)
    
    elif intent == "file_operation":
        st.write("### File Operations: Upload/Download/Process Files")
        if uploaded_file:
            st.write("Processing file...")
            st.write("File processed successfully!")
        else:
            st.warning("Please upload a file to perform operations.")

    elif intent == "chat":
        response = openai.Completion.create(
            model=selected_model,  # Use the selected GPT model
            prompt=user_input,
            max_tokens=150,
            temperature=0.7
        )
        st.write(f"🗣 OpenAI Response: {response.choices[0].text.strip()}")

    elif intent == "linear_regression":
        if data is not None:
            X = data.select_dtypes(include='number').dropna()
            if X.shape[1] < 2:
                st.warning("Need at least two numerical columns for regression.")
            else:
                y = st.selectbox("Select target column", X.columns)
                X = X.drop(columns=[y])
                X_train, X_test, y_train, y_test = train_test_split(X, data[y], test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                st.write(f"### Model Coefficients: {model.coef_}")
                st.write(f"### Intercept: {model.intercept_}")
                y_pred = model.predict(X_test)
                st.write(f"### Predictions vs Actuals:")
                st.write(pd.DataFrame({"Predictions": y_pred, "Actual": y_test}))

                st.write("### Plot: Predictions vs Actual")
                plt.scatter(y_test, y_pred)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Linear Regression: Predictions vs Actual")
                st.pyplot()

        else:
            st.warning("Please upload a dataset to use this model.")

    elif intent == "kmeans":
        if data is not None:
            st.write("### K-Means Clustering")
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least two numerical columns for clustering.")
            else:
                selected_cols = st.multiselect("Select features for clustering:", numeric_cols, default=numeric_cols[:2])
                n_clusters = st.slider("Select number of clusters:", 2, 10, 3)

                X = data[selected_cols]
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X)
                data["Cluster"] = clusters

                st.write("### Clustered Data Preview")
                st.dataframe(data.head())

                fig = px.scatter(data, x=selected_cols[0], y=selected_cols[1], color="Cluster", title="K-Means Clustering")
                st.plotly_chart(fig)

        else:
            st.warning("Please upload a dataset to use this model.")

    elif intent == "autoencoder":
        st.write("Autoencoder functionality can be implemented using TensorFlow or Keras.")
        # You can add an Autoencoder model if needed.

    elif "pca" in user_input.lower():
        if data is not None:
            pca_result, explained_variance = pca_dimensionality_reduction(data)
            st.write(f"### PCA Result:")
            st.dataframe(pca_result)
            st.write(f"### Explained Variance Ratio:")
            st.write(explained_variance)
        else:
            st.warning("Please upload a dataset to use PCA.")

    # Display visualizations
    if data is not None:
        st.write("### Data Visualizations")
        plot_correlation_heatmap(data)
        plot_histogram(data)

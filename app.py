import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    # Ensure same columns as training
    df_encoded = pd.get_dummies(df)

    # Add missing columns
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep correct order
    df_encoded = df_encoded[model_columns]

    # Prediction
    predictions = model.predict(df_encoded)

    df["Predicted_Churn"] = predictions

    st.subheader("Prediction Results")
    st.write(df)

    # Download predictions
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions CSV",
        csv,
        "churn_predictions.csv",
        "text/csv"
    )
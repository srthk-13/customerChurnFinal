import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# ---------- CUSTOM STYLING ----------
st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #d3cce3, #e9e4f0);
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1532619675605-1b4a2f930c1f");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .big-title {
            font-size: 3rem !important;
            color: #4B0082;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4B0082;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- MODEL LOADING ----------
model = pickle.load(open("churn_model.pkl", "rb"))
trained_columns = pickle.load(open("model_columns.pkl", "rb"))

# ---------- UI ----------
st.markdown('<div class="big-title">🔮 Churn Prediction App</div>', unsafe_allow_html=True)
st.markdown("Give us some customer details, and we’ll predict whether they’re staying or leaving. ✨")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Predict button
if st.button("🚀 Predict Now"):
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [1 if senior == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }

    input_df = pd.DataFrame(input_data)

    # Label encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    label_maps = {
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'PaperlessBilling': ['Yes', 'No']
    }
    for col in binary_cols:
        le = LabelEncoder()
        le.fit(label_maps[col])
        input_df[col] = le.transform(input_df[col])

    # One-hot encoding
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                      'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols, drop_first=True)

    # Align
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("🎉 This customer is likely to stay.")
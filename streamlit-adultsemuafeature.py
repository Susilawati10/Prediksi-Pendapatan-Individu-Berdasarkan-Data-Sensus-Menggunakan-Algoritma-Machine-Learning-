import streamlit as st
import joblib
import numpy as np

# ================= LOAD MODEL DAN SCALER =================
model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# ================= CONFIGURASI HALAMAN =================
st.set_page_config(page_title="Income Prediction", layout="centered")
st.title("Income Prediction App using XGBoost")
st.markdown("Please fill in the information below to predict income category.")

# ================= INPUT UTAMA =================
age = st.number_input("Age", min_value=17, max_value=100, value=30)
education = st.selectbox("Education Level", [
    'Low-Education', 'HS-grad', 'Some-college', 'Associate',
    'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
])
marital_status = st.selectbox("Marital Status", [
    'Never-married', 'Married-civ-spouse', 'Divorced', 'Other'
])
occupation = st.selectbox("Occupation", [
    'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
    'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing',
    'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv',
    'Armed-Forces', 'Priv-house-serv'
])
relationship = st.selectbox("Relationship", [
    'Not-in-family', 'Husband', 'Own-child', 'Other-relative'
])
race = st.selectbox("Race", [
    'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
])
sex = st.radio("Sex", ['Male', 'Female'])
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
native_country = st.selectbox("Native Country", [
    'United-States', 'Mexico', 'Philippines', 'Germany', 'India', 'Other'
])

# ================= FITUR TAMBAHAN =================
net_capital = st.number_input("Net Capital (gain - loss)", value=0)
is_overtime = st.selectbox("Working Overtime?", [0, 1])
family_size = st.slider("Family Size", 0, 20, value=1)
is_self_employed = st.selectbox("Self-employed?", [0, 1])

# ================= ENCODING =================
sex = 1 if sex == "Male" else 0

native_country_map = {
    'United-States': 0.26, 'Mexico': 0.11, 'Philippines': 0.34,
    'Germany': 0.31, 'India': 0.48, 'Other': 0.15
}
occupation_map = {
    'Exec-managerial': 0.48, 'Prof-specialty': 0.46, 'Tech-support': 0.42,
    'Priv-house-serv': 0.10, 'Other-service': 0.12, 'Sales': 0.28,
    'Craft-repair': 0.23, 'Handlers-cleaners': 0.13, 'Machine-op-inspct': 0.16,
    'Transport-moving': 0.20, 'Farming-fishing': 0.14,
    'Protective-serv': 0.30, 'Armed-Forces': 0.50, 'Adm-clerical': 0.21
}
native_country_enc = native_country_map.get(native_country, 0.15)
occupation_enc = occupation_map.get(occupation, 0.20)

education_map = {
    'Low-Education': 0, 'HS-grad': 1, 'Some-college': 2, 'Associate': 3,
    'Bachelors': 4, 'Masters': 5, 'Prof-school': 6, 'Doctorate': 7
}
education = education_map[education]

# One-hot encoding
marital_ohe = [
    int(marital_status == 'Married-civ-spouse'),
    int(marital_status == 'Divorced'),
    int(marital_status == 'Other')
]
relationship_ohe = [
    int(relationship == 'Husband'),
    int(relationship == 'Not-in-family'),
    int(relationship == 'Other-relative'),
    int(relationship == 'Own-child')
]
race_ohe = [
    int(race == 'Amer-Indian-Eskimo'),
    int(race == 'Asian-Pac-Islander'),
    int(race == 'Black'),
    int(race == 'Other'),
    int(race == 'White')
]

# Gabungkan semua fitur → total 22 kolom
input_data = np.array([[age, education, hours_per_week, sex, net_capital,
                        native_country_enc, occupation_enc, is_overtime,
                        family_size, is_self_employed] +
                       marital_ohe + relationship_ohe + race_ohe])

# ================= PREDIKSI =================
if input_data.shape[1] != 22:
    st.error(f"Feature count mismatch: got {input_data.shape[1]}, expected 22.")
else:
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Income"):
        prediction = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0][prediction]
        result_label = "<=50K" if prediction == 0 else ">50K"

        # Tampilan hasil
        if prediction == 1:
            st.success(f"Prediction: {prediction} — Income Class: {result_label} (Probability: {proba:.2f})")
        else:
            st.warning(f"Prediction: {prediction} — Income Class: {result_label} (Probability: {proba:.2f})")

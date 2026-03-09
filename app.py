import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from reportlab.pdfgen import canvas

# configure page
st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️", layout="wide")

# load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

# load dataset
df = pd.read_csv("heart.csv")

# encode categorical columns for visualization
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Machine Learning Pipeline using Logistic Regression")

# sidebar inputs
st.sidebar.header("Enter Patient Data")

age = st.sidebar.slider("Age", 20, 100, 40)

sex = st.sidebar.selectbox("Sex", ["M","F"])

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA","NAP","ASY","TA"]
)

resting_bp = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)

cholesterol = st.sidebar.slider("Cholesterol", 100, 600, 200)

fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0,1])

rest_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal","ST","LVH"]
)

max_hr = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)

exercise_angina = st.sidebar.selectbox(
    "Exercise Angina",
    ["Y","N"]
)

oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)

# convert categorical values
sex = 1 if sex == "M" else 0
exercise_angina = 1 if exercise_angina == "Y" else 0

chest_pain_map = {"ATA":0,"NAP":1,"ASY":2,"TA":3}
rest_ecg_map = {"Normal":0,"ST":1,"LVH":2}
st_slope_map = {"Up":0,"Flat":1,"Down":2}

chest_pain = chest_pain_map[chest_pain]
rest_ecg = rest_ecg_map[rest_ecg]
st_slope = st_slope_map[st_slope]

st.subheader("Heart Disease Prediction")

# input data must match training features
input_data = np.array([[
age,
sex,
chest_pain,
resting_bp,
cholesterol,
fasting_bs,
rest_ecg,
max_hr,
exercise_angina,
oldpeak,
st_slope
]])

if st.button("Predict Risk"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    with col2:
        st.metric("Risk Probability", f"{probability*100:.2f}%")

    # risk gauge
    st.subheader("Heart Risk Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    # create pdf report
    report_file = "heart_report.pdf"
    c = canvas.Canvas(report_file)

    c.drawString(100,800,"Heart Disease Prediction Report")
    c.drawString(100,760,f"Age: {age}")
    c.drawString(100,740,f"Sex: {sex}")
    c.drawString(100,720,f"Chest Pain Type: {chest_pain}")
    c.drawString(100,700,f"Resting BP: {resting_bp}")
    c.drawString(100,680,f"Cholesterol: {cholesterol}")
    c.drawString(100,660,f"Max HR: {max_hr}")
    c.drawString(100,640,f"Risk Probability: {probability*100:.2f}%")

    c.save()

    with open(report_file,"rb") as file:
        st.download_button(
            label="Download Patient Report",
            data=file,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )

# dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# visualizations
st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    sns.countplot(x="HeartDisease", data=df)
    plt.title("Heart Disease Distribution")
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    sns.histplot(df["Age"], kde=True)
    plt.title("Age Distribution")
    st.pyplot(fig2)

# correlation heatmap
st.subheader("Feature Correlation")

numeric_df = df.select_dtypes(include=["int64","float64"])

fig3 = plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig3)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ LOAD MODEL ------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoder = joblib.load("encoder.pkl")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Burnout Predictor", page_icon="🔥", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🔥 Developer Burnout Prediction System")
st.write("Predict burnout level based on work and lifestyle factors")
result_placeholder = st.empty()

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("📌 Enter Your Details")
age = st.sidebar.slider("Age", 20, 44, 25)

experience_years = st.sidebar.slider(
    "Experience Years", 0, 19, 2
)

daily_work_hours = st.sidebar.slider(
    "Daily Work Hours", 4, 14, 8
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours", 4, 9, 7
)

caffeine_intake = st.sidebar.slider(
    "Caffeine Intake", 0, 7, 2
)

bugs_per_day = st.sidebar.slider(
    "Bugs Per Day", 0, 19, 3
)

commits_per_day = st.sidebar.slider(
    "Commits Per Day", 0, 29, 8
)

meetings_per_day = st.sidebar.slider(
    "Meetings Per Day", 0, 9, 2
)

screen_time = st.sidebar.slider(
    "Screen Time", 5.0, 19.0, 8.0
)

exercise_hours = st.sidebar.slider(
    "Exercise Hours", 0.0, 2.0, 1.0
)

stress_level = st.sidebar.slider(
    "Stress Level", 0, 100, 50
)

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Your Input Summary")
    st.write(f"Age: {age}")
    st.write(f"Work Hours: {daily_work_hours}")
    st.write(f"Sleep Hours: {sleep_hours}")
    st.write(f"Stress Level: {stress_level}")

with col2:
    st.subheader("📈 Lifestyle Overview")
    fig, ax = plt.subplots()
    ax.bar(
        ["Work", "Sleep", "Stress"],
        [daily_work_hours, sleep_hours, stress_level]
    )
    st.pyplot(fig)
if st.button("🚀 Predict Burnout"):

    user_data = pd.DataFrame([{
        "age": age,
        "experience_years": experience_years,
        "daily_work_hours": daily_work_hours,
        "sleep_hours": sleep_hours,
        "caffeine_intake": caffeine_intake,
        "bugs_per_day": bugs_per_day,
        "commits_per_day": commits_per_day,
        "meetings_per_day": meetings_per_day,
        "screen_time": screen_time,
        "exercise_hours": exercise_hours,
        "stress_level": stress_level
    }])

    user_imputed = imputer.transform(user_data)
    user_scaled = scaler.transform(user_imputed)

    prediction = model.predict(user_scaled)
    result = str(encoder.inverse_transform(prediction)[0])

    if result == "High":
        st.error("🔥 High Burnout Risk - Take rest!")
    elif result == "Medium":
        st.warning("⚠️ Medium Burnout Risk - Manage workload")
    else:
        st.success("✅ Low Burnout Risk - You're fine")

    probs = model.predict_proba(user_scaled)[0]

    st.subheader("📊 Prediction Confidence")

    for cls, prob in zip(encoder.classes_, probs):
        st.write(f"**{cls}:** {prob:.2%}")

    st.info("💡 Tip: More sleep and less stress reduce burnout risk.")

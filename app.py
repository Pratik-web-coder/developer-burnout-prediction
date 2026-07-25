import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Burnout Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, imputer, encoder

model, scaler, imputer, encoder = load_artifacts()

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
    :root {
        --accent: #FF4B4B;
        --accent-soft: #FFECEC;
        --ink: #1F2430;
        --muted: #6B7280;
    }

    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }

    /* Header banner */
    .hero {
        padding: 2rem 2.25rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #1F2430 0%, #33394A 100%);
        color: white;
        margin-bottom: 1.75rem;
        box-shadow: 0 8px 24px rgba(31, 36, 48, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #C9CDD6;
        font-size: 0.95rem;
    }

    /* Section cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid #EAECEF;
        margin-bottom: 1.2rem;
    }
    .card h3 {
        margin-top: 0;
        font-size: 1.05rem;
        color: var(--ink);
    }

    /* Metric chips */
    .chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .chip {
        background: #F1F5F9;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        color: var(--ink);
        border: 1px solid #E2E8F0;
    }

    /* Predict button */
    div.stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        background: #E23C3C;
        transform: translateY(-1px);
    }

    /* Result banners */
    .result-box {
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }
    .result-high   { background: #FEECEC; color: #B42318; border: 1px solid #FDA29B; }
    .result-medium { background: #FFF6E5; color: #93500B; border: 1px solid #FEC84B; }
    .result-low    { background: #ECFDF3; color: #027A48; border: 1px solid #A6F4C5; }

    footer {visibility: hidden;}
    .app-footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🔥 Developer Burnout Prediction System</h1>
    <p>A machine-learning tool that estimates burnout risk from work habits, lifestyle, and stress signals.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR INPUTS — grouped into logical sections
# =========================================================
st.sidebar.markdown("## 📌 Your Profile")

with st.sidebar.expander("👤 Personal & Experience", expanded=True):
    age = st.slider("Age", 20, 44, 25)
    experience_years = st.slider("Experience (years)", 0, 19, 2)

with st.sidebar.expander("💻 Work Patterns", expanded=True):
    daily_work_hours = st.slider("Daily Work Hours", 4, 14, 8)
    commits_per_day = st.slider("Commits Per Day", 0, 29, 8)
    bugs_per_day = st.slider("Bugs Per Day", 0, 19, 3)
    meetings_per_day = st.slider("Meetings Per Day", 0, 9, 2)

with st.sidebar.expander("🌙 Lifestyle & Wellbeing", expanded=True):
    sleep_hours = st.slider("Sleep Hours", 4, 9, 7)
    caffeine_intake = st.slider("Caffeine Intake (cups)", 0, 7, 2)
    screen_time = st.slider("Screen Time (hrs)", 5.0, 19.0, 8.0)
    exercise_hours = st.slider("Exercise Hours", 0.0, 2.0, 1.0)
    stress_level = st.slider("Stress Level", 0, 100, 50)

predict_clicked = st.sidebar.button("🚀 Predict Burnout Risk")

# =========================================================
# MAIN LAYOUT — SUMMARY + LIFESTYLE GAUGE
# =========================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card"><h3>📊 Input Summary</h3>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", age)
    m2.metric("Work Hrs/Day", daily_work_hours)
    m3.metric("Sleep Hrs", sleep_hours)
    m4.metric("Stress", f"{stress_level}/100")

    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    st.markdown(f"""
        <span class="chip">☕ {caffeine_intake} cups/day</span>
        <span class="chip">🖥️ {screen_time}h screen time</span>
        <span class="chip">🏃 {exercise_hours}h exercise</span>
        <span class="chip">🐛 {bugs_per_day} bugs/day</span>
        <span class="chip">✅ {commits_per_day} commits/day</span>
        <span class="chip">📅 {meetings_per_day} meetings/day</span>
    """, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>🌡️ Stress Gauge</h3>', unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress_level,
        number={'suffix': " / 100"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, 40], 'color': "#ECFDF3"},
                {'range': [40, 70], 'color': "#FFF6E5"},
                {'range': [70, 100], 'color': "#FEECEC"},
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
if predict_clicked:
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
        "stress_level": stress_level,
    }])

    user_imputed = imputer.transform(user_data)
    user_scaled = scaler.transform(user_imputed)

    prediction = model.predict(user_scaled)
    result = str(encoder.inverse_transform(prediction)[0])
    probs = model.predict_proba(user_scaled)[0]

    st.markdown("### 🧠 Prediction Result")

    result_class = {"High": "result-high", "Medium": "result-medium", "Low": "result-low"}.get(result, "result-low")
    result_icon = {"High": "🔥", "Medium": "⚠️", "Low": "✅"}.get(result, "✅")
    result_msg = {
        "High": "High Burnout Risk — consider taking rest and reducing workload.",
        "Medium": "Medium Burnout Risk — keep an eye on workload and recovery time.",
        "Low": "Low Burnout Risk — your current balance looks healthy.",
    }.get(result, "")

    st.markdown(f'<div class="result-box {result_class}">{result_icon} {result_msg}</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="card"><h3>📈 Prediction Confidence</h3>', unsafe_allow_html=True)
        for cls, prob in sorted(zip(encoder.classes_, probs), key=lambda x: -x[1]):
            st.markdown(
                f'<p style="color:#1F2430; font-weight:600; margin:0.6rem 0 0.2rem 0;">{cls}</p>',
                unsafe_allow_html=True,
            )
            st.progress(float(prob))
            st.markdown(
                f'<p style="color:#6B7280; font-size:0.85rem; margin:0.1rem 0 0 0;">{prob:.1%}</p>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="card"><h3>💡 Recommendations</h3>', unsafe_allow_html=True)
        tips = []
        if sleep_hours < 6:
            tips.append("Aim for at least 7 hours of sleep — you're currently under-resting.")
        if stress_level > 65:
            tips.append("Stress is elevated. Consider short breaks or workload redistribution.")
        if exercise_hours < 0.5:
            tips.append("Light daily exercise (even 20–30 min) measurably lowers reported burnout.")
        if screen_time > 10:
            tips.append("Screen time is high — scheduled breaks can reduce fatigue.")
        if not tips:
            tips.append("Your habits look balanced. Keep maintaining this routine.")
        tips_html = "".join(
            f'<p style="color:#1F2430; margin:0.5rem 0; line-height:1.4;">• {t}</p>' for t in tips
        )
        st.markdown(tips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👈 Adjust your details in the sidebar, then click **Predict Burnout Risk** to see results.")

# =========================================================
# FOOTER
# =========================================================
st.markdown('<div class="app-footer">Developer Burnout Prediction System · Built with Streamlit & Scikit-learn</div>', unsafe_allow_html=True)

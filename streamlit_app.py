import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model/job_success_model.pkl")
le_edu = joblib.load("model/le_edu.pkl")
le_applied = joblib.load("model/le_applied.pkl")
le_cover = joblib.load("model/le_cover.pkl")

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title="Job Success Predictor", layout="centered")

# ----------------------- HEADER -----------------------
st.markdown("<h1 style='text-align: center; color: teal;'>💼 Job Application Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("#### 📋 Enter your profile details below to check your chances of being shortlisted.")

st.divider()

# ----------------------- INPUT FIELDS -----------------------
col1, col2 = st.columns(2)

with col1:
    experience = st.slider("👨‍💻 Years of Experience", 0, 20, 2)
    skills_match = st.slider("🧠 Skills Match (1 to 5)", 1, 5, 3)
    certifications = st.slider("🎓 Certifications", 0, 5, 1)

with col2:
    education = st.selectbox("📘 Education Level", ["High School", "Bachelor’s", "Master’s", "PhD"])
    applied_before = st.radio("📄 Applied Before?", ["Yes", "No"], horizontal=True)
    cover_letter = st.radio("✉️ Cover Letter Included?", ["Yes", "No"], horizontal=True)

st.divider()

# ----------------------- PREDICT BUTTON -----------------------
if st.button("🎯 Predict"):
    input_data = np.array([
        experience,
        le_edu.transform([education])[0],
        skills_match,
        certifications,
        le_applied.transform([applied_before])[0],
        le_cover.transform([cover_letter])[0]
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.divider()

    if prediction == 1:
        st.success(f"✅ You are likely to be shortlisted! \n\nConfidence: **{probability:.2f}%**")
    else:
        st.error(f"❌ Less likely to be shortlisted. \n\nConfidence: **{probability:.2f}%**")

    st.markdown("---")
    st.markdown("🔁 *You can tweak your profile details and try again!*")

# ----------------------- FOOTER -----------------------
st.markdown("<br><hr style='border: 1px solid #ccc'><div style='text-align:center'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

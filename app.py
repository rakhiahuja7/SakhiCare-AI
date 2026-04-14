import pandas as pd
import os
import plotly.express as px
import streamlit as st
import pickle
from utils import triage_layer
from image_utils import predict_skin_condition

st.set_page_config(
    page_title="SakhiCare AI",
    page_icon="🌸",
    layout="centered"
)

# -------------------- STYLING --------------------
st.markdown("""
<style>
/* ===== Main App ===== */
.stApp {
    background-color: #fff5f7;
    color: #1f2937;
}

/* ===== Titles ===== */
h1, h2, h3 {
    color: #c2185b !important;
    font-weight: 700;
}

/* ===== Labels ===== */
label, p, span {
    color: #1f2937 !important;
}

/* ===== Text area ===== */
textarea {
    background-color: white !important;
    color: #111827 !important;
    border: 2px solid #f8bbd0 !important;
    border-radius: 12px !important;
}

textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

/* ===== Number input ===== */
input {
    background-color: white !important;
    color: #111827 !important;
}

/* ===== Selectbox ===== */
[data-baseweb="select"] > div {
    background-color: white !important;
    color: #111827 !important;
}

/* selected text */
[data-baseweb="select"] span {
    color: #111827 !important;
}

/* dropdown popup */
div[role="listbox"] {
    background-color: white !important;
}

/* dropdown items */
div[role="option"] {
    background-color: white !important;
    color: #111827 !important;
}

/* hover option */
div[role="option"]:hover {
    background-color: #fce7f3 !important;
    color: #c2185b !important;
}

/* ===== File uploader box ===== */
/* ===== File uploader full fix ===== */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border: 2px dashed #f8bbd0 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* inner upload section */
[data-testid="stFileUploader"] section {
    background-color: white !important;
    color: #111827 !important;
}

/* drag-drop area */
[data-testid="stFileUploaderDropzone"] {
    background-color: white !important;
    color: #111827 !important;
    border: 1px solid #f8bbd0 !important;
}

/* upload button */
[data-testid="stFileUploader"] button {
    background-color: white !important;
    color: #111827 !important;
    border: 1px solid #f8bbd0 !important;
}

/* all uploader text */
[data-testid="stFileUploader"] * {
    color: #111827 !important;
    background-color: white !important;
    opacity: 1 !important;
}

/* ===== Main Analyze button ===== */
.stButton > button {
    background-color: #ec4899 !important;
    color: white !important;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #db2777 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
with open("sakhi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["symptom", "prediction"]).to_csv(HISTORY_FILE, index=False)

# -------------------- HEADER --------------------
st.title("🌸 SakhiCare AI")
st.subheader("Your private women’s health companion 💜")
st.write("Describe your symptoms privately and get AI-powered wellness guidance.")

st.markdown("---")

# -------------------- INPUT --------------------
st.header("💬 Tell me what you are feeling")
symptom_input = st.text_area(
    "Describe your symptoms",
    placeholder="Example: white discharge with itching, acne flare-ups and mild cramps"
)

uploaded_image = st.file_uploader(
    "📸 Upload face / neck skin image (optional)",
    type=["jpg", "jpeg", "png"]
)

# -------------------- EXTRA PERSONALIZATION --------------------
st.header("📋 Extra Personalization")
age = st.number_input("Age", 10, 60, 22)
cycle_delay = st.slider("Cycle delay (days)", 0, 30, 0)
pain_level = st.slider("Pain level", 0, 10, 0)

discharge_color = st.selectbox(
    "Discharge color",
    ["None", "White", "Brown", "Yellow", "Red"]
)

itching = st.radio("Any itching?", ["No", "Yes"])
pregnancy_chance = st.radio("Pregnancy possibility?", ["No", "Yes"])

st.markdown("---")

# -------------------- ANALYSIS --------------------
if st.button("🔍 Analyze Symptoms"):
    if symptom_input.strip() == "":
        st.warning("Please describe your symptoms first.")
    else:
        # NLP prediction
        vectorized_input = vectorizer.transform([symptom_input])
        prediction = model.predict(vectorized_input)[0]

        # rule-based triage
        triage_result = triage_layer(
            symptom_input,
            pain_level,
            cycle_delay,
            discharge_color,
            itching,
            pregnancy_chance
        )

        st.success("🌸 Analysis Complete")

        # Friend mode response
        st.markdown("### 💜 Sakhi Friend Mode")
        st.write(
            f"Hey girl 💕 based on what you shared, this may relate to **{prediction.upper()}**. "
            "Please don’t stress — many women face similar issues, and awareness is the first step 💖"
        )

        # NLP output
        st.subheader("🧠 NLP Symptom Category")
        st.info(prediction.upper())

        # CNN image analysis
        if uploaded_image is not None:
            st.subheader("📸 Visual Hormonal Insight")
            st.image(uploaded_image, width=250)

            img_pred, confidence = predict_skin_condition(uploaded_image)

            st.success(
                f"Possible skin pattern detected: **{img_pred.upper()}** "
                f"({confidence:.2f}% confidence)"
            )
        else:
            st.info("No skin image uploaded. Visual analysis skipped.")

        # urgency
        st.subheader("🚦 Urgency Guidance")
        st.write(triage_result)

        # self care
        st.subheader("💡 Basic Self-Care Tips")
        if prediction == "pcos":
            st.write("- Track your cycle regularly")
            st.write("- Manage stress and sleep")
            st.write("- Reduce sugar intake")
            st.write("- Consider hormonal profile tests")
        elif prediction == "infection":
            st.write("- Maintain intimate hygiene")
            st.write("- Wear breathable cotton underwear")
            st.write("- Avoid scented washes")
        elif prediction == "pregnancy_risk":
            st.write("- Take a pregnancy test")
            st.write("- Track missed cycle")
            st.write("- Consult gynecologist if unsure")
        else:
            st.write("- Stay hydrated")
            st.write("- Monitor symptoms")
            st.write("- Seek doctor help if persistent")

        # disclaimer
        st.subheader("⚠️ Disclaimer")
        st.caption(
            "This AI provides early wellness guidance only and is not a replacement for professional medical advice."
        )

        # save history
        history_df = pd.read_csv(HISTORY_FILE)
        new_row = pd.DataFrame({
            "symptom": [symptom_input],
            "prediction": [prediction]
        })
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        history_df.to_csv(HISTORY_FILE, index=False)

# -------------------- HISTORY --------------------
st.markdown("---")
st.header("📊 Symptom History Insights")

history_df = pd.read_csv(HISTORY_FILE)

if not history_df.empty:
    fig = px.histogram(
        history_df,
        x="prediction",
        title="Prediction Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No symptom history yet.")

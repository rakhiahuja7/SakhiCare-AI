import pandas as pd
import os
import plotly.express as px
import streamlit as st
import pickle
from utils import triage_layer

st.set_page_config(
    page_title="SakhiCare AI",
    page_icon="🌸",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff0f5;
        color: #2d1b3d;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #6a1b9a !important;
        font-weight: 700;
    }

    p, label, div, span {
        color: #2d1b3d !important;
    }

    .stTextArea textarea {
        background-color: #1e1e2f;
        color: white;
        border-radius: 12px;
    }

    .stNumberInput input {
        background-color: #1e1e2f;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with open("sakhi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["symptom", "prediction"]).to_csv(HISTORY_FILE, index=False)

st.title("🌸 SakhiCare AI")
st.subheader("Your private women’s health companion 💜")
st.write(
    "Describe your symptoms privately and get AI-powered health guidance."
)

st.markdown("---")

st.header("💬 Tell me what you are feeling")
symptom_input = st.text_area(
    "Describe your symptoms",
    placeholder="Example: white discharge with itching and mild cramps"
)


st.header("📋 Extra Personalization")
age = st.number_input("Age", min_value=10, max_value=60, value=22)

cycle_delay = st.slider("Cycle delay (days)", 0, 30, 0)

pain_level = st.slider("Pain level", 0, 10, 0)

discharge_color = st.selectbox(
    "Discharge color",
    ["None", "White", "Brown", "Yellow", "Red"]
)

itching = st.radio("Any itching?", ["No", "Yes"])

pregnancy_chance = st.radio("Pregnancy possibility?", ["No", "Yes"])

st.markdown("---")

if st.button("🔍 Analyze Symptoms"):
    if symptom_input.strip() == "":
        st.warning("Please describe your symptoms first.")
    else:
        vectorized_input = vectorizer.transform([symptom_input])
        prediction = model.predict(vectorized_input)[0]

        history_df = pd.read_csv(HISTORY_FILE)
        new_row = pd.DataFrame({
            "symptom": [symptom_input],
            "prediction": [prediction]
        })
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        history_df.to_csv(HISTORY_FILE, index=False)

        triage_result = triage_layer(symptom_input)

        st.success("🌸 Analysis Complete")

        st.subheader("🧠 Possible Category")
        st.info(prediction.upper())

        st.subheader("🚦 Urgency Guidance")
        st.write(triage_result)

        st.subheader("💡 Basic Self-Care Tips")
        if prediction == "pcos":
            st.write("- Track your menstrual cycle")
            st.write("- Reduce stress")
            st.write("- Maintain healthy diet and sleep")
        elif prediction == "infection":
            st.write("- Maintain hygiene")
            st.write("- Wear cotton underwear")
            st.write("- Avoid harsh soaps")
        elif prediction == "pregnancy":
            st.write("- Take a pregnancy test")
            st.write("- Stay hydrated")
            st.write("- Consult OB-GYN if symptoms persist")
        else:
            st.write("- Monitor symptoms")
            st.write("- Rest well")
            st.write("- Stay hydrated")

        st.subheader("⚠️ Disclaimer")
        st.caption(
            "This AI provides guidance only and is not a substitute for professional medical advice."
        )

st.markdown("---")
st.header("📊 Symptom History Insights")

history_df = pd.read_csv(HISTORY_FILE)

if not history_df.empty:
    fig = px.histogram(
        history_df,
        x="prediction",
        title="Prediction Frequency"
    )
    st.plotly_chart(fig)
else:
    st.info("No history available yet.")

def triage_layer(symptom_text, pain_level=0, cycle_delay=0,
                 discharge_color="None", itching="No",
                 pregnancy_chance="No"):

    text = symptom_text.lower()

    if pain_level >= 8:
        return "🔴 High pain detected. Please consult a gynecologist immediately."

    if pregnancy_chance == "Yes" and cycle_delay >= 7:
        return "🟠 Possible pregnancy risk. Please take a pregnancy test and consult a doctor."

    if discharge_color in ["Yellow", "Red"] and itching == "Yes":
        return "🟠 Signs of possible infection. Please consult a doctor soon."

    if "pcos" in text or cycle_delay > 15:
        return "🟡 Possible PCOS or hormonal imbalance symptoms."

    return "🟢 Mild symptoms. Self-care and monitoring should be okay for now."
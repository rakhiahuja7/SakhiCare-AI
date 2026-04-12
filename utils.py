def triage_layer(symptom_text):
    text = symptom_text.lower()

    urgent_keywords = [
        "heavy bleeding",
        "severe cramps",
        "pelvic pain",
        "fever",
        "foul smell",
        "pregnancy pain"
    ]

    doctor_keywords = [
        "white discharge",
        "itching",
        "missed period",
        "acne",
        "facial hair",
        "irregular cycle"
    ]

    for word in urgent_keywords:
        if word in text:
            return "🔴 Urgent: Please seek medical attention immediately."

    for word in doctor_keywords:
        if word in text:
            return "🟡 Doctor Soon: Please consult a gynecologist in 1–2 days."

    return "🟢 Self-Care: Monitor symptoms, rest well, and stay hydrated."
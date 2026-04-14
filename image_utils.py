import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained CNN model
cnn_model = load_model("skin_cnn_model.keras")

# IMPORTANT: Must match dataset folder names EXACTLY
class_names = [
    "acne_dataset",
    "dark_spots",
    "inflammatory_acne",
    "non_inflammatory_acne_blackheads",
    "non_inflammatory_acne_whiteheads",
    "pigmentation",
    "pores",
    "redness",
    "skin_v2",
    "wrinkles"
]

def predict_skin_condition(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_array)

    pred_index = int(np.argmax(prediction[0]))

    # Safety fallback
    if pred_index >= len(class_names):
        return "unknown_skin_condition", 0.0

    predicted_class = class_names[pred_index]
    confidence = float(np.max(prediction[0]))

    return predicted_class, confidence
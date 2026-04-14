import pickle
import tensorflow as tf

# ---------------- TEST NLP ----------------
with open("sakhi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

sample_symptom = [
    "white discharge with itching and lower abdominal cramps"
]

sample_vec = vectorizer.transform(sample_symptom)
prediction = model.predict(sample_vec)

print("🌸 NLP Prediction:", prediction[0])

# ---------------- TEST CNN ----------------
cnn_model = tf.keras.models.load_model("skin_cnn_model.keras")

print("📸 CNN model loaded successfully")
print("Input shape:", cnn_model.input_shape)
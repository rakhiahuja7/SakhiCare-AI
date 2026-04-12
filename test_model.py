import pickle
from utils import triage_layer

# Load saved files
with open("sakhi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# User symptom input
user_input = input("Describe your symptom: ")

# Predict
vectorized_input = vectorizer.transform([user_input])
prediction = model.predict(vectorized_input)[0]

# Triage result
triage_result = triage_layer(user_input)

print("\n🌸 SakhiCare AI Result 🌸")
print("Possible category:", prediction)
print("Advice level:", triage_result)
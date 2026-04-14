import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------- LOAD DATA ----------------
df = pd.read_csv("symptoms_dataset.csv")

# expected columns:
# symptom_text, category
X = df["symptom_text"]
y = df["category"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- VECTORIZATION ----------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------- EVALUATION ----------------
preds = model.predict(X_test_vec)

print("✅ NLP MODEL REPORT")
print(classification_report(y_test, preds))

# ---------------- SAVE ----------------
with open("sakhi_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("sakhi_model.pkl saved")
print("vectorizer.pkl saved")
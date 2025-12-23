import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset (USE YOUR EXACT FILE NAME)
df = pd.read_csv("data/customer_support_tickets.csv")

# Combine text columns
df["text"] = (
    df["Ticket Subject"].astype(str) + " " +
    df["Ticket Description"].astype(str)
)

X_text = df["text"]
y = df["Ticket Type"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X = vectorizer.fit_transform(X_text)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully")

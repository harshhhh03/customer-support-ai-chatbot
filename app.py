from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pickle
import numpy as np

# App setup
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

CONFIDENCE_THRESHOLD = 0.45

FALLBACK_REPLY = (
    "Sorry, I can only help with customer support issues "
    "like orders, refunds, and service requests."
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(message: str):
    msg = message.lower().strip()

    # 1️⃣ RULE-BASED GREETINGS (IMPROVED)
    if any(greet in msg for greet in ["hi", "hello", "hey", "good morning", "good evening"]):
        return {
            "reply": "Hello! Please describe your issue related to orders, refunds, or support.",
            "confidence": "Rule-based"
        }

    # 2️⃣ KEYWORD SHORTCUTS (IMPORTANT)
    if any(word in msg for word in ["refund", "money back", "return"]):
        return {
            "reply": "I understand this is a refund-related issue. Our support team will help you with this.",
            "confidence": "High (keyword)"
        }

    if any(word in msg for word in ["order", "delivery", "shipment", "track"]):
        return {
            "reply": "This seems related to an order issue. Please share your order details.",
            "confidence": "High (keyword)"
        }

    # 3️⃣ ML MODEL (FOR COMPLEX QUERIES)
    data = vectorizer.transform([message])
    probs = model.predict_proba(data)[0]
    max_prob = float(np.max(probs))

    if max_prob < CONFIDENCE_THRESHOLD:
        return {
            "reply": FALLBACK_REPLY,
            "confidence": "Low"
        }

    intent = model.classes_[np.argmax(probs)]
    confidence = round(max_prob * 100, 2)

    return {
        "reply": f"I understand this issue is related to {intent}. Our support team can assist you.",
        "confidence": f"{confidence}%"
    }

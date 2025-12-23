# 🤖 Customer Support AI Chatbot

A production-ready **AI-powered customer support chatbot** built using a real-world customer support ticket dataset.  
The chatbot uses a **hybrid approach (Rule-based + Machine Learning)** to understand and respond to user queries.

## 🚀 Live Demo

👉 **https://customer-support-ai-chatbot-dj8a.onrender.com**

## ✨ Features

- Hybrid chatbot (Rule-based + ML)
- Intent classification using **TF-IDF + Logistic Regression**
- Trained on real customer support ticket data
- Modern chat UI with auto dark mode (system-based)
- Confidence-based responses
- FastAPI backend
- Deployed live on Render

## 🧠 How It Works

### 1️⃣ Rule-Based Layer
- Handles greetings (`hi`, `hello`)
- Detects keywords like `refund`, `order`, `delivery`
- Provides fast and reliable responses

### 2️⃣ Machine Learning Layer
- Text vectorization using TF-IDF
- Logistic Regression classifier
- Predicts ticket type based on user input

### 3️⃣ Fallback Handling
- Safely responds to unrelated queries

## 🗂️ Project Structure

customer-support-ai-chatbot/
│
├── app.py # FastAPI application
├── train.py # ML training script
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│ └── index.html # Chat UI
│
├── static/
│ └── style.css # Styling (light + dark mode)
│
├── data/
│ └── customer_support_tickets.csv
## 🧪 Model Details

- **Features**:
  - Ticket Subject
  - Ticket Description
- **Label**:
  - Ticket Type
- **Algorithm**:
  - Logistic Regression
- **Vectorizer**:
  - TF-IDF

## 🛠️ Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- HTML, CSS, JavaScript
- Render (Deployment)

## ▶️ Run Locally

pip install -r requirements.txt
python train.py
uvicorn app:app --reload
Open browser:

http://127.0.0.1:8000

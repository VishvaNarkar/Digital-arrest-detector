import streamlit as st
import joblib
import re

# Load model + vectorizer
model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Risky keywords (extend this list as needed)
risky_keywords = ["bank", "account", "verify", "locked", "password",
                  "urgent", "winner", "lottery", "click", "payment",
                  "prize", "free", "offer", "limited", "risk", "security",
                  "immediately", "suspend", "alert", "confirm",
                  "transaction", "access", "details", "information",
                  "identity", "social", "security", "ssn", "credit", "debit"]

st.title("📧 Digital Arrest Detector (Hybrid Scam Detection)")
st.write("This tool uses **ML + Keyword Matching** for stronger scam detection.")

# Input box
user_input = st.text_area("Paste an email or SMS text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Step 1: ML Model Prediction
        X_input = vectorizer.transform([user_input])
        scam_prob = model.predict_proba(X_input)[0][1]

        # Step 2: Keyword Matching
        found_keywords = [kw for kw in risky_keywords if re.search(rf"\b{kw}\b", user_input, re.IGNORECASE)]
        keyword_score = len(found_keywords)

        # Step 3: Hybrid Decision Logic
        if scam_prob > 0.3 or keyword_score >= 2:
            label = "🚨 Likely Scam"
        else:
            label = "✅ Likely Safe"

        # Output results
        st.subheader("🔍 Analysis Result")
        st.write(f"**Classification:** {label}")
        st.write(f"**ML Scam Probability:** {scam_prob:.2f}")
        st.write(f"**Risky Keywords Found:** {', '.join(found_keywords) if found_keywords else 'None'}")

        # Highlight risky words in text
        highlighted_text = user_input
        for kw in found_keywords:
            highlighted_text = re.sub(
                rf"(?i)\b({kw})\b",
                r"**[\1]**",
                highlighted_text
            )

        st.subheader("📄 Highlighted Message")
        st.markdown(highlighted_text)

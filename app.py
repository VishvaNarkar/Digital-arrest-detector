import streamlit as st
import joblib
import re

# --- Load model + vectorizer ---
clf = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Suspicious keyword list for highlighting
SUSPICIOUS_KEYWORDS = [
    "urgent", "arrest", "police", "court", "pay", "fine", "transfer",
    "otp", "bank", "freeze", "imprisonment", "lawsuit", "verify"
]

def highlight_suspicious(text):
    """Highlight risky keywords in the input text."""
    highlighted = text
    for word in SUSPICIOUS_KEYWORDS:
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        highlighted = pattern.sub(f"**:red[{word}]**", highlighted)
    return highlighted

# --- Streamlit UI ---
st.title("🚨 Digital Arrest Scam Detector (Text Channel Prototype)")
st.write("Paste an email, SMS, or chat message below and detect if it's a possible scam.")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        X_vec = vectorizer.transform([user_input])
        proba = clf.predict_proba(X_vec)[0][1]  # scam probability

        st.subheader("Result:")
        if proba > 0.7:
            st.error(f"⚠️ High Scam Risk! (Score: {proba:.2f})")
        elif proba > 0.4:
            st.warning(f"❓ Suspicious - Medium Risk (Score: {proba:.2f})")
        else:
            st.success(f"✅ Likely Safe (Score: {proba:.2f})")

        st.markdown("### Highlighted text with risky words:")
        st.write(highlight_suspicious(user_input))

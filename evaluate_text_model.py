import pandas as pd
from sklearn.metrics import classification_report
from app import detect_message, text_model, vectorizer

# Load dataset (replace with your path)
df = pd.read_csv("data/sms_spam.csv")  
df.columns = ["label", "text"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

y_true = []
y_pred_ml = []
y_pred_hybrid = []

for _, row in df.iterrows():
    text = str(row["text"])
    
    y_true.append(row["label_num"])

    # ML-only
    X_input = vectorizer.transform([text])
    ml_prob = text_model.predict_proba(X_input)[0][1]
    y_pred_ml.append(1 if ml_prob > 0.5 else 0)

    # Hybrid
    result = detect_message(text)
    y_pred_hybrid.append(1 if "Scam" in result["label"] else 0)

print("\n=== ML-Only Model ===")
print(classification_report(y_true, y_pred_ml))

print("\n=== Hybrid (ML + Sentiment + Keywords) ===")
print(classification_report(y_true, y_pred_hybrid))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

# --- Load dataset (SMS Spam Collection as example) ---
# Dataset format: 2 columns [label, message]
# label: 'spam' or 'ham', message: text
df = pd.read_csv("data/sms_spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]

# Binary labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["text"]
y = df["label"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TF-IDF + Logistic Regression ---
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
clf = LogisticRegression(max_iter=1000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# --- Save model + vectorizer ---
joblib.dump(clf, "models/text_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved.")

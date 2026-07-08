import os
import re
import json
import joblib
import nltk
import datetime
import pandas as pd
from pathlib import Path
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords


def clean_text(text: str) -> str:
    """Lowercase, remove URLs and punctuation, and normalize spaces.

    Preserves Unicode word characters AND Unicode combining marks so that
    multilingual content — Hindi/Gujarati script and OTP digits — is not
    silently discarded, which would hurt both training and scam detection.

    Background: Python's ``\\w`` matches Unicode letters and digits but does
    NOT match Unicode combining marks (category Mc/Mn), such as Devanagari
    matras (e.g. ी U+0940) or Gujarati matras (e.g. ી U+0AC0).  These
    diacritics are glued to consonants to form vowel sounds; stripping them
    fragments every Indic word into meaningless pieces.  The explicit block
    ranges below add them back to the keep-set:

      U+0300–U+036F  Latin combining diacritics
      U+0900–U+097F  Devanagari (Hindi)
      U+0A80–U+0AFF  Gujarati
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Keep: Unicode word chars (\w = letters/digits/_), whitespace, and
    # Unicode combining marks for Indic scripts.
    _KEEP = r"\w\s\u0300-\u036f\u0900-\u097f\u0a80-\u0aff"
    text = re.sub(rf"[^{_KEEP}]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save_model(data_path: Path, model_dir: Path):
    """Train SMS spam classifier and save model, vectorizer, metadata, and LLM dataset."""

    # --- Setup ---
    nltk.download("stopwords", quiet=True)
    stop_words = stopwords.words("english")
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    df = pd.read_csv(data_path, encoding="latin-1")

    # Normalize column names if necessary
    if "text" not in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v2": "text", "v1": "label"})

    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)

    # Convert labels to numeric
    df["label_numeric"] = df["label"].map({"ham": 0, "spam": 1})

    # --- Balance dataset (upsample minority class) ---
    ham = df[df.label == "ham"].sample(frac=1, random_state=42)
    spam = df[df.label == "spam"].sample(frac=1, random_state=42)
    spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
    df_balanced = pd.concat([ham, spam_upsampled], ignore_index=True)

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced["clean_text"],
        df_balanced["label_numeric"],
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # --- TF-IDF Vectorization ---
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Train model ---
    clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test_tfidf)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    # --- Save model and vectorizer ---
    joblib.dump(clf, model_dir / "text_model.pkl")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    print("\nModel and vectorizer saved successfully.")

    # --- Save metadata ---
    metadata = {
        "trained_at": datetime.datetime.now().isoformat(),
        "accuracy": float(acc),
        "model": "LogisticRegression (TF-IDF, balanced)",
        "vectorizer_params": vectorizer.get_params(),
        "data_samples": int(len(df_balanced)),
    }

    # Save metadata in a JSON-safe way (convert non-serializable objects to strings)
    def _make_json_safe(obj):
        # numpy / pandas objects often implement .tolist(); use that when available
        try:
            if hasattr(obj, "tolist"):
                return obj.tolist()
        except Exception:
            pass
        # fallback to string representation
        return str(obj)

    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            # convert nested dict values
            safe_metadata[k] = {kk: _make_json_safe(vv) for kk, vv in v.items()}
        else:
            safe_metadata[k] = _make_json_safe(v)

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
    print("Metadata saved.")

    # --- Create LLM fine-tuning dataset ---
    df_balanced["label_text"] = df_balanced["label_numeric"].map({0: "ham", 1: "spam"})
    instruction_text = "Classify the message as spam or not spam."
    df_llm = pd.DataFrame({
        "instruction": [instruction_text] * len(df_balanced),
        "input": df_balanced["clean_text"].tolist(),
        "output": df_balanced["label_text"].tolist(),
    })

    df_llm.to_json(model_dir / "sms_spam_llm.jsonl", orient="records", lines=True)
    train_llm, test_llm = train_test_split(df_llm, test_size=0.2, random_state=42, shuffle=True)
    train_llm.to_json(model_dir / "train.jsonl", orient="records", lines=True)
    test_llm.to_json(model_dir / "test.jsonl", orient="records", lines=True)
    print("LLM datasets (train/test) saved.")


def main():
    """Main entry point for training the spam detector.

    Paths are derived relative to this script file so the script is
    portable across operating systems and CI environments.
    """
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "sms_spam.csv"
    model_dir = base_dir / "models"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Make sure 'data/sms_spam.csv' exists relative to this script."
        )

    train_and_save_model(data_path, model_dir)
    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()

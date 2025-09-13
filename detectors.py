# detectors.py
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# predefine keyword categories
URGENCY_KEYWORDS = {"urgent","immediately","now","asap","right away","today"}
THREAT_KEYWORDS = {"arrest","jail","fine","lawsuit","penalty","detain","prosecute"}
MONEY_KEYWORDS = {"transfer","pay","bank","account","card","ssn","otp","cvv","payment","wire","loan"}
IDENTITY_KEYWORDS = {"verify","confirm","identity","details","information","login","credentials"}

analyzer = SentimentIntensityAnalyzer()

def keyword_features(text, keyword_sets=None):
    if keyword_sets is None:
        keyword_sets = [URGENCY_KEYWORDS, THREAT_KEYWORDS, MONEY_KEYWORDS, IDENTITY_KEYWORDS]
    found = {}
    t = text.lower()
    for idx, kws in enumerate(keyword_sets):
        count = sum(1 for w in kws if re.search(rf"\b{re.escape(w)}\b", t))
        found[idx] = count
    return found  # dict: {0: urgency_count, 1: threat_count, ...}

def sentiment_urgency_score(text):
    vs = analyzer.polarity_scores(text)
    # VADER returns pos/neu/neg/compound. For urgency/fear we look at negative + compound absolute
    neg = vs["neg"]
    compound = vs["compound"]
    # heuristic: negative + strong compound (abs) => higher risk
    score = min(1.0, neg*2 + max(0, -compound))  
    return score  # 0..1

def text_pattern_score(text, text_model, vectorizer):
    """Return ML prob + keyword counts + urgency score"""
    ml_prob = float(text_model.predict_proba(vectorizer.transform([text]))[0][1])
    kws = keyword_features(text)
    urgency = sentiment_urgency_score(text)
    return {"ml_prob": ml_prob, "urgency": urgency, "kws": kws}

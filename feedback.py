# feedback.py
import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "data/feedback.csv"

def append_feedback(source, content, predicted_label, user_label, metadata=None):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "content": content,
        "predicted": predicted_label,
        "user_label": user_label
    }
    if metadata:
        row.update(metadata)
    df = pd.DataFrame([row])
    try:
        df.to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)
    except Exception as e:
        print("Feedback save error:", e)

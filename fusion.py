# fusion.py
def fuse_scores(text_score, audio_prob=None, video_prob=None, reputation_score=0.0, weights=None):
    # text_score is dict with 'ml_prob' and 'urgency'
    if weights is None:
        weights = {"text":0.55, "audio":0.2, "video":0.2, "reputation":0.05}
    fused = 0.0
    fused += weights["text"] * (0.6*text_score["ml_prob"] + 0.4*text_score["urgency"])
    if audio_prob is not None:
        fused += weights["audio"] * audio_prob
    if video_prob is not None:
        fused += weights["video"] * video_prob
    fused += weights["reputation"] * reputation_score
    # normalize in case of missing channels
    total_w = sum(weights.values())
    fused = fused / total_w
    return float(fused)

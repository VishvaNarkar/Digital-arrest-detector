"""URL and domain name analyzer for phishing and scam link detection.

Examines extracted URLs for suspicious patterns, brand mismatch, IP addresses,
and high-risk Top-Level Domains (TLDs).
"""
import re
from urllib.parse import urlparse

# --- Brand official domain mappings (lowercased) ---
OFFICIAL_DOMAINS = {
    "hdfc": "hdfcbank.com",
    "sbi": "sbi.co.in",
    "icici": "icicibank.com",
    "paytm": "paytm.com",
    "netflix": "netflix.com",
    "amazon": "amazon.com",
    "fedex": "fedex.com",
    "dhl": "dhl.com",
    "trai": "trai.gov.in",
    "uidai": "uidai.gov.in",
}

# --- Suspicious or commonly abused TLDs ---
SUSPICIOUS_TLDS = {
    "xyz", "top", "cc", "click", "club", "info", "net", "org", "work", "bid",
    "loan", "gq", "cf", "tk", "ml", "ga"
}

# --- Abused URL shorteners ---
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "shorturl.at", "is.gd", "wa.link"}


def extract_urls(text: str) -> list[str]:
    """Find and return all URLs inside the given text using regex."""
    if not text:
        return []
    # Match standard URL schemas or www. subdomains
    pattern = r"(https?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*)"
    return re.findall(pattern, text, re.IGNORECASE)


def analyze_url(url: str, text: str) -> dict:
    """Analyze a single URL for scam and phishing indicators.

    Parameters
    ----------
    url : str
        The full URL string to analyze.
    text : str
        The original text of the message containing the URL (used for brand checks).

    Returns
    -------
    dict
        A dict containing:
        - risk_score: float (0.0 to 1.0)
        - flags: list[str] (e.g., "[URL] Brand Impersonation")
    """
    flags = []
    risk_score = 0.0

    # Clean URL and parse domain
    if not url.lower().startswith("http"):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    try:
        parsed = urlparse(url_to_parse)
        netloc = parsed.netloc.lower()
        # Remove port number if any
        hostname = netloc.split(":")[0] if ":" in netloc else netloc
    except Exception:
        return {"risk_score": 0.5, "flags": ["[URL] Invalid URL Format"]}

    if not hostname:
        return {"risk_score": 0.0, "flags": []}

    # 1. Direct IP Address check
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    if re.match(ip_pattern, hostname):
        flags.append("[URL] Direct IP Address")
        risk_score = max(risk_score, 0.9)

    # 2. Link Shortener check
    if hostname in SHORTENERS or any(hostname.endswith("." + s) for s in SHORTENERS):
        flags.append("[URL] Abused Link Shortener")
        risk_score = max(risk_score, 0.4)

    # 3. Top-Level Domain (TLD) check
    tld = hostname.split(".")[-1] if "." in hostname else ""
    if tld in SUSPICIOUS_TLDS:
        # Check if the text contains high-profile keywords
        for brand in OFFICIAL_DOMAINS.keys():
            if brand in text.lower():
                flags.append(f"[URL] Suspicious TLD ({tld}) with brand mention")
                risk_score = max(risk_score, 0.85)
                break
        else:
            flags.append(f"[URL] Suspicious TLD ({tld})")
            risk_score = max(risk_score, 0.3)

    # 4. Brand Mismatch / Domain Impersonation Check
    # (e.g. text mentions 'HDFC' or 'FedEx' but link host is not hdfcbank.com/fedex.com)
    for brand, official_domain in OFFICIAL_DOMAINS.items():
        if brand in text.lower():
            # If the hostname doesn't match the official domain, flag it
            if official_domain not in hostname:
                # But make sure the hostname contains the brand name as a substring (impersonation)
                if brand in hostname:
                    flags.append(f"[URL] Brand Impersonation ({brand})")
                    risk_score = max(risk_score, 0.9)
                # Or just general mismatch when requesting verification/login
                elif any(kw in text.lower() for kw in ["verify", "login", "otp", "suspend", "kyc"]):
                    flags.append(f"[URL] Domain Brand Mismatch ({brand})")
                    risk_score = max(risk_score, 0.8)

    # 5. Phishing Subdomain Keywords Check (e.g. otp.secure-login-hdfc.com)
    phishing_keywords = ["otp", "login", "verify", "secure", "kyc", "bank", "account", "refund"]
    parts = hostname.split(".")
    # Skip the TLD and SLD if standard
    domain_subparts = parts[:-2] if len(parts) > 2 else parts[:-1]
    for subpart in domain_subparts:
        for kw in phishing_keywords:
            if kw in subpart and not any(official in hostname for official in OFFICIAL_DOMAINS.values()):
                flags.append(f"[URL] Suspicious Phishing Keyword in Domain ({kw})")
                risk_score = max(risk_score, 0.6)
                break

    return {"risk_score": risk_score, "flags": flags}

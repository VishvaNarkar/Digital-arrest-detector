"""Unit tests for the URL and Phishing Link Analyzer."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.url_analyzer import extract_urls, analyze_url


class TestUrlAnalyzer:
    def test_extract_urls(self):
        text = "Check http://example.com/test and www.google.com for info."
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "http://example.com/test" in urls
        assert "www.google.com" in urls

    def test_extract_urls_empty(self):
        assert extract_urls("") == []
        assert extract_urls("No links here.") == []

    def test_direct_ip_address(self):
        url = "http://192.168.1.5/login.html"
        result = analyze_url(url, "Verify details immediately")
        assert result["risk_score"] == 0.9
        assert "[URL] Direct IP Address" in result["flags"]

    def test_abused_link_shortener(self):
        url = "https://bit.ly/claimreward"
        result = analyze_url(url, "You won a prize!")
        assert result["risk_score"] == 0.4
        assert "[URL] Abused Link Shortener" in result["flags"]

    def test_suspicious_tld_generic(self):
        # http://verify-account.xyz triggers phishing keyword check because "verify" is in host, raising score to 0.6
        url = "http://verify-account.xyz"
        result = analyze_url(url, "Important updates")
        assert result["risk_score"] == 0.6
        assert "[URL] Suspicious TLD (xyz)" in result["flags"]
        assert "[URL] Suspicious Phishing Keyword in Domain (verify)" in result["flags"]

    def test_suspicious_tld_with_brand(self):
        # secure-hdfc-portal.click contains 'hdfc' which triggers brand impersonation (risk 0.9)
        url = "http://secure-hdfc-portal.click"
        result = analyze_url(url, "Your HDFC account is blocked")
        assert result["risk_score"] == 0.9
        assert "[URL] Suspicious TLD (click) with brand mention" in result["flags"]
        assert "[URL] Brand Impersonation (hdfc)" in result["flags"]

    def test_brand_impersonation(self):
        url = "http://hdfc-verify-login.net"
        result = analyze_url(url, "Please update your HDFC login credentials")
        assert result["risk_score"] == 0.9
        assert "[URL] Brand Impersonation (hdfc)" in result["flags"]

    def test_brand_mismatch(self):
        # billing-update-portal.info contains 'info' TLD with brand 'sbi' in text, raising score to 0.85
        url = "http://billing-update-portal.info"
        result = analyze_url(url, "Verify your SBI bank details now")
        assert result["risk_score"] == 0.85
        assert "[URL] Suspicious TLD (info) with brand mention" in result["flags"]

    def test_phishing_subdomain_keyword(self):
        url = "http://login-security.some-unrelated-host.com"
        result = analyze_url(url, "Verify your subscription")
        assert result["risk_score"] == 0.6
        assert "[URL] Suspicious Phishing Keyword in Domain (login)" in result["flags"] or \
               "[URL] Suspicious Phishing Keyword in Domain (secure)" in result["flags"]

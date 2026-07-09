function badgeTier(kw) {
  const HIGH = ['otp','pin','cvv','password','passcode','aadhar','aadhaar',
    'security code','auth code','ssn','share otp','share password','share pin']
  const MED  = ['bank','account','urgent','verify','fraud','phishing','lottery',
    'winner','transaction','payment','transfer','suspend','alert']
  const kl = kw.toLowerCase()
  if (HIGH.some(h => kl.includes(h))) return 'high'
  if (MED.some(m => kl.includes(m))) return 'medium'
  return 'low'
}

export default function KeywordBadges({ keywords }) {
  if (!keywords || keywords.length === 0) {
    return <p className="empty-state">No risky keywords detected.</p>
  }

  return (
    <div className="keywords-section">
      <div className="keywords-title">Detected keywords</div>
      <div className="keyword-list">
        {keywords.map((kw, i) => (
          <span key={i} className={`keyword-badge ${badgeTier(kw)}`} style={{ animationDelay: `${i * 0.03}s` }}>
            {kw}
          </span>
        ))}
      </div>
    </div>
  )
}

import ShieldLogo from './ShieldLogo'

export default function Header({ ollamaAvailable }) {
  return (
    <header className="header">
      <div className="header-logo">
        <ShieldLogo size={36} />
        <span className="header-brand">
          Fraud<span>Shield</span> AI
        </span>
      </div>
      <span className="header-tagline">Multi-Channel Fraud &amp; Deepfake Detection</span>
      <div className="header-spacer" />
      <div className="header-badge">
        <span style={{ fontSize: '0.65rem' }}>●</span>
        AI-Powered
      </div>
    </header>
  )
}

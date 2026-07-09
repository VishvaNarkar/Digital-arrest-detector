import ShieldLogo from './ShieldLogo'

export default function Header({ ollamaAvailable, threshold, onToggleSidebar, sidebarOpen }) {
  return (
    <header className="header">
      {/* Mobile Sidebar Hamburger Toggle */}
      <button
        type="button"
        className="menu-toggle-btn"
        onClick={onToggleSidebar}
        aria-label="Toggle sidebar history"
        aria-expanded={sidebarOpen}
      >
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
        </svg>
      </button>

      <div className="header-branding">
        <ShieldLogo size={32} />
        <div>
          <div className="header-brand">FraudShield AI</div>
          <div className="header-tagline">Multi-Channel Scam & Deepfake Intelligence</div>
        </div>
      </div>

      <div className="header-spacer" />

      <div className="header-meta">
        <span className="header-chip">Threshold {(threshold * 100).toFixed(0)}%</span>
        <span className={`header-status ${ollamaAvailable ? 'online' : 'offline'}`}>
          <span className="header-status-dot" />
          {ollamaAvailable ? 'Reasoning Ready' : 'Heuristic Mode'}
        </span>
      </div>
    </header>
  )
}

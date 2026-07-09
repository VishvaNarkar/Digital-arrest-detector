export default function Sidebar({
  sessions = [],
  activeSessionId = null,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClearAll,
  threshold,
  onThresholdChange,
  ollamaAvailable,
  sidebarOpen,
  onCloseSidebar
}) {
  const sliderPct = `${((threshold - 0.1) / 0.7) * 100}%`

  return (
    <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
      {/* Sidebar Header with New Chat Button */}
      <div className="sidebar-header">
        <button type="button" className="new-chat-btn" onClick={() => {
          onNewSession()
          if (onCloseSidebar) onCloseSidebar()
        }}>
          <span className="plus-icon" style={{ display: 'flex', alignItems: 'center' }}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          </span> 
          New analysis
        </button>
      </div>

      {/* Chat Session History List */}
      <div className="sidebar-sessions-section">
        <div className="sidebar-section-title">Analysis History</div>
        {sessions.length === 0 ? (
          <div className="sessions-empty-state">
            No past analyses.
          </div>
        ) : (
          <div className="sessions-list">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`session-item-wrapper ${session.id === activeSessionId ? 'active' : ''}`}
              >
                <button
                  type="button"
                  className="session-item-btn"
                  onClick={() => {
                    onSelectSession(session.id)
                    if (onCloseSidebar) onCloseSidebar()
                  }}
                  title={session.title}
                >
                  <span className="session-icon" style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)' }}>
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                  </span>
                  <span className="session-title">{session.title}</span>
                </button>
                <button
                  type="button"
                  className="session-delete-btn"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(session.id)
                  }}
                  title="Delete chat"
                >
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Sidebar Settings and System Status */}
      <div className="sidebar-footer">
        {/* Risk Threshold slider */}
        <div className="sidebar-footer-card">
          <div className="footer-card-header">
            <strong>Risk Sensitivity</strong>
            <span className="slider-value-pill">{(threshold * 100).toFixed(0)}%</span>
          </div>
          <p className="footer-card-description">Lower catches more suspicious activity. Higher reduces noise.</p>
          <input
            type="range"
            className="slider"
            min="0.10"
            max="0.80"
            step="0.05"
            value={threshold}
            onChange={e => onThresholdChange(parseFloat(e.target.value))}
            style={{ '--slider-pct': sliderPct }}
            aria-label="Risk detection threshold"
            id="threshold-slider"
          />
          <div className="slider-hints">
            <span>Higher recall</span>
            <span>Lower noise</span>
          </div>
        </div>

        {/* System Status online/offline indicator */}
        <div className="sidebar-footer-card soft-bg">
          <div className={`status-row ${ollamaAvailable ? 'online' : 'offline'}`}>
            <span className="status-bullet" />
            <div>
              <strong>{ollamaAvailable ? 'Reasoning Engine Online' : 'Heuristic Mode Only'}</strong>
              <p>{ollamaAvailable ? 'LLM-assisted explanation is available.' : 'Deep learning & rule engines active.'}</p>
            </div>
          </div>
        </div>

        {/* Action button to clear all chats */}
        {sessions.length > 0 && (
          <button
            type="button"
            className="clear-history-btn"
            onClick={onClearAll}
          >
            Clear all analyses
          </button>
        )}
      </div>
    </aside>
  )
}

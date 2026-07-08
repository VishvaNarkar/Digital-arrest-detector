import { useState, useEffect } from 'react'

export default function Sidebar({ threshold, onThresholdChange, ollamaAvailable }) {
  // Update CSS custom property for the slider gradient fill
  const sliderPct = ((threshold - 0.10) / (0.80 - 0.10) * 100).toFixed(1) + '%'

  return (
    <aside className="sidebar">
      {/* Detection Sensitivity */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Detection Sensitivity</div>
        <div className="threshold-value">
          <span className="threshold-label">Scam threshold</span>
          <span className="threshold-pill">{(threshold * 100).toFixed(0)}%</span>
        </div>
        <input
          type="range"
          className="slider"
          min="0.10"
          max="0.80"
          step="0.05"
          value={threshold}
          onChange={e => onThresholdChange(parseFloat(e.target.value))}
          style={{ '--slider-pct': sliderPct }}
          aria-label="Scam detection threshold"
          id="threshold-slider"
        />
        <div className="slider-hints">
          <span>Catch more</span>
          <span>Fewer false positives</span>
        </div>
      </div>

      {/* Stats */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Detection Stats</div>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">96%</div>
            <div className="stat-label">Accuracy</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">300+</div>
            <div className="stat-label">Keywords</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">3</div>
            <div className="stat-label">Languages</div>
          </div>
        </div>
      </div>

      {/* Ollama status */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">LLM Status</div>
        <div className="ollama-status">
          <span className={`status-dot ${ollamaAvailable ? 'online' : 'offline'}`} />
          <span>
            {ollamaAvailable
              ? 'Ollama online (Phi3:mini)'
              : 'Ollama offline — heuristic mode'}
          </span>
        </div>
      </div>

      {/* How it works */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">How It Works</div>
        <div className="how-steps">
          {[
            'Select a channel: Text, Audio, or Video',
            'Upload or type your content',
            'Get instant AI-powered results',
          ].map((text, i) => (
            <div className="how-step" key={i}>
              <div className="step-num">{i + 1}</div>
              <div className="step-text">{text}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Supported languages */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Languages</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { flag: '', name: 'Indian English (en-in)' },
            { flag: '', name: 'Hindi (hi)' },
            { flag: '', name: 'Gujarati (gu)' },
          ].map(({ flag, name }) => (
            <div key={name} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: 8 }}>
              <span>{flag}</span><span>{name}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}

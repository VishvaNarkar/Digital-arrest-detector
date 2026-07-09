import ResultCard from './common/ResultCard'

export default function MessageBubble({ message }) {
  const { sender, text, file, result, loading, error, isAudio, isVideo } = message

  return (
    <div className={`message-row ${sender}`}>
      <div className="message-avatar">
        {sender === 'user' ? (
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
        )}
      </div>
      <div className="message-bubble-wrapper">
        {sender === 'assistant' && (
          <div className="message-sender-name">FraudShield AI</div>
        )}
        {sender === 'user' && (
          <div className="message-sender-name">You</div>
        )}
        <div className="message-bubble-content">
          {/* Render User Attachments */}
          {file && (
            <div className="message-file-card">
              <span className="file-icon" style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center' }}>
                {isAudio ? (
                  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 18V5l12-2v13"/>
                    <circle cx="6" cy="18" r="3"/>
                    <circle cx="18" cy="16" r="3"/>
                  </svg>
                ) : isVideo ? (
                  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 7l-7 5 7 5V7z"/>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                )}
              </span>
              <div className="file-info">
                <span className="file-name">{file.name}</span>
                <span className="file-meta">
                  {isAudio ? 'Audio file' : isVideo ? 'Video file' : 'File'} • {Math.round((file.size / 1024 / 1024) * 100) / 100} MB
                </span>
              </div>
            </div>
          )}

          {/* Render Text Content */}
          {text && <div className="message-text">{text}</div>}

          {/* Render Loading State */}
          {loading && (
            <div className="message-loading">
              <div className="typing-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
              <span className="loading-text">
                {isAudio
                  ? 'Transcribing speech and detecting fraud signals...'
                  : isVideo
                  ? 'Decoding frames and verifying video authenticity...'
                  : 'Analyzing message content and scanning databases...'}
              </span>
            </div>
          )}

          {/* Render Error State */}
          {error && (
            <div className="message-error-card">
              <div className="error-icon" style={{ display: 'flex', alignItems: 'center' }}>
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                  <line x1="12" y1="9" x2="12" y2="13"/>
                  <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
              </div>
              <div className="error-content">
                <strong>Analysis Failed</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          {/* Render Result Cards */}
          {result && !loading && !error && (
            <div className="message-result-container animate-slide-up">
              {/* If it's a video result, show video custom messages */}
              {isVideo ? (
                <>
                  <ResultCard result={result} />
                  {result.label?.includes('Deepfake') ? (
                    <div className="alert danger-light spaced border-radius-md" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                      </svg>
                      <div>
                        <strong>High Risk:</strong> This video shows signs of face/voice manipulation. Verify the source before sharing.
                      </div>
                    </div>
                  ) : (
                    <div className="alert success-light spaced border-radius-md" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                      <div>
                        <strong>Low Risk:</strong> No significant deepfake indicators detected.
                      </div>
                    </div>
                  )}
                </>
              ) : isAudio ? (
                /* Audio results return combined results: result.analysis + result.transcription */
                <ResultCard result={result.analysis} transcription={result.transcription} />
              ) : (
                /* Text results are normal */
                <ResultCard result={result} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

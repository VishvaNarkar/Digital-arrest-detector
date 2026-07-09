import { useRef, useEffect } from 'react'

export default function ChatInput({
  input,
  setInput,
  selectedFile,
  setSelectedFile,
  lang,
  setLang,
  onSubmit,
  loading
}) {
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  // Auto-expand textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!loading && (input.trim() || selectedFile)) {
        onSubmit()
      }
    }
  }

  const isAudio = selectedFile?.type?.startsWith('audio/') || selectedFile?.name?.endsWith('.wav') || selectedFile?.name?.endsWith('.mp3')
  const isVideo = selectedFile?.type?.startsWith('video/') || selectedFile?.name?.endsWith('.mp4') || selectedFile?.name?.endsWith('.avi') || selectedFile?.name?.endsWith('.mov')

  return (
    <div className="chat-input-container">
      {selectedFile && (
        <div className="file-preview-bar">
          <div className="file-preview-card">
            <span className="file-preview-icon" style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center' }}>
              {isAudio ? (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 18V5l12-2v13"/>
                  <circle cx="6" cy="18" r="3"/>
                  <circle cx="18" cy="16" r="3"/>
                </svg>
              ) : isVideo ? (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 7l-7 5 7 5V7z"/>
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
              )}
            </span>
            <div className="file-preview-info">
              <span className="file-preview-name">{selectedFile.name}</span>
              <span className="file-preview-size">
                {Math.round((selectedFile.size / 1024 / 1024) * 100) / 100} MB
              </span>
            </div>
            <button
              type="button"
              className="file-preview-remove"
              onClick={() => {
                setSelectedFile(null)
                if (fileInputRef.current) fileInputRef.current.value = ''
              }}
              title="Remove file"
            >
              &times;
            </button>
          </div>

          {isAudio && (
            <div className="input-lang-selector animate-fade-in">
              <label htmlFor="input-lang">Transcribe in:</label>
              <select
                id="input-lang"
                value={lang}
                onChange={e => setLang(e.target.value)}
                disabled={loading}
              >
                <option value="en-in">Indian English</option>
                <option value="hi">Hindi (हिंदी)</option>
                <option value="gu">Gujarati (ગુજરાતી)</option>
              </select>
            </div>
          )}
        </div>
      )}

      <div className="chat-input-row">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="audio/*,video/*,.wav,.mp3,.mp4,.avi,.mov"
          style={{ display: 'none' }}
          id="chat-file-upload"
        />

        <button
          type="button"
          className={`attachment-btn ${selectedFile ? 'has-file' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
          title="Attach audio or video file"
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>

        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={1}
          placeholder={
            selectedFile
              ? `Add a message or press enter to analyze ${selectedFile.name}...`
              : "Paste spam text here, or attach an audio/video file..."
          }
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />

        <button
          type="button"
          className="send-btn"
          onClick={onSubmit}
          disabled={loading || (!input.trim() && !selectedFile)}
          title="Send message for analysis"
        >
          {loading ? (
            <div className="spinner-sm" />
          ) : (
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="3">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
            </svg>
          )}
        </button>
      </div>
      <div className="chat-input-footer-note">
        FraudShield AI scans communications for security indicators. Verify high-stakes transactions.
      </div>
    </div>
  )
}

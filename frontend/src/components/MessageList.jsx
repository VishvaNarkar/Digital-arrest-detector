import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

const STARTER_PROMPTS = [
  {
    icon: (
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
    ),
    label: 'Suspicious OTP Warning',
    text: 'Your bank account will be suspended unless you verify your OTP passcode immediately here: http://scam-bank.com',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 12 20 22 4 22 4 12"/>
        <rect x="2" y="7" width="20" height="5"/>
        <line x1="12" y1="22" x2="12" y2="7"/>
        <path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/>
        <path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/>
      </svg>
    ),
    label: 'Lottery Winner Notification',
    text: 'Congratulations! You won a cash prize of $50,000. Send your card number and pin to claim it.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 18V5l12-2v13"/>
        <circle cx="6" cy="18" r="3"/>
        <circle cx="18" cy="16" r="3"/>
      </svg>
    ),
    label: 'Audio Fraud Analysis',
    text: '',
    info: 'Attach an audio file (WAV/MP3) and select transcription language to scan spoken words.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M23 7l-7 5 7 5V7z"/>
        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
      </svg>
    ),
    label: 'Deepfake Verification',
    text: '',
    info: 'Attach a video file (MP4/AVI/MOV) to sample frames and estimate facial manipulation probability.',
  }
]

export default function MessageList({ messages, onStarterClick }) {
  const bottomRef = useRef(null)

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="chat-welcome-container">
        <div className="welcome-header">
          <div className="welcome-logo" style={{ color: 'var(--primary)' }}>
            <svg viewBox="0 0 24 24" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <h1>How can FraudShield assist you today?</h1>
          <p>
            Secure, multi-channel scam intelligence. Paste a suspicious message, email, or upload call recordings and video clips to analyze fraud signals instantly.
          </p>
        </div>

        <div className="starter-grid">
          {STARTER_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              className="starter-card"
              onClick={() => onStarterClick(prompt)}
            >
              <span className="starter-icon" style={{ color: 'var(--primary)' }}>
                {prompt.icon}
              </span>
              <div className="starter-text-content">
                <strong>{prompt.label}</strong>
                <p>{prompt.text || prompt.info}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="messages-list">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div ref={bottomRef} style={{ height: 1 }} />
    </div>
  )
}

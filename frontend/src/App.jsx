import { useEffect, useState, useMemo } from 'react'
import Header from './components/Layout/Header'
import Sidebar from './components/Layout/Sidebar'
import MessageList from './components/MessageList'
import ChatInput from './components/ChatInput'
import { getHealth, analyzeText, analyzeAudio, analyzeVideo } from './api/client'

export default function App() {
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem('fraudshield_sessions')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  
  const [activeSessionId, setActiveSessionId] = useState(() => {
    try {
      return localStorage.getItem('fraudshield_active_id') || null
    } catch {
      return null
    }
  })

  const [threshold, setThreshold] = useState(() => {
    try {
      const saved = localStorage.getItem('fraudshield_threshold')
      return saved ? parseFloat(saved) : 0.35
    } catch {
      return 0.35
    }
  })

  const [ollamaAvailable, setOllamaAvailable] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Current session input state (temporary, not saved in history until sent)
  const [input, setInput] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [lang, setLang] = useState('en-in')
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  // Save sessions to localStorage
  useEffect(() => {
    localStorage.setItem('fraudshield_sessions', JSON.stringify(sessions))
  }, [sessions])

  // Save active session ID to localStorage
  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem('fraudshield_active_id', activeSessionId)
    } else {
      localStorage.removeItem('fraudshield_active_id')
    }
  }, [activeSessionId])

  // Save threshold to localStorage
  useEffect(() => {
    localStorage.setItem('fraudshield_threshold', String(threshold))
  }, [threshold])

  // System health monitoring
  useEffect(() => {
    let mounted = true

    async function checkHealth() {
      try {
        const health = await getHealth()
        if (mounted) setOllamaAvailable(Boolean(health.ollama_available))
      } catch {
        if (mounted) setOllamaAvailable(false)
      }
    }

    checkHealth()
    const intervalId = setInterval(checkHealth, 30_000)

    return () => {
      mounted = false
      clearInterval(intervalId)
    }
  }, [])

  // Auto-initialize first session if empty
  useEffect(() => {
    if (sessions.length === 0) {
      handleNewSession()
    } else if (!activeSessionId) {
      setActiveSessionId(sessions[0].id)
    }
  }, [sessions, activeSessionId])

  const activeSession = useMemo(() => {
    return sessions.find(s => s.id === activeSessionId) || sessions[0] || null
  }, [sessions, activeSessionId])

  const messages = useMemo(() => {
    return activeSession ? activeSession.messages : []
  }, [activeSession])

  function handleNewSession() {
    const newId = `session_${Date.now()}`
    const newSession = {
      id: newId,
      title: 'New Analysis',
      messages: [],
      createdAt: new Date().toISOString()
    }
    setSessions(prev => [newSession, ...prev])
    setActiveSessionId(newId)
    setInput('')
    setSelectedFile(null)
  }

  function handleSelectSession(id) {
    setActiveSessionId(id)
    setInput('')
    setSelectedFile(null)
  }

  function handleDeleteSession(id) {
    const updated = sessions.filter(s => s.id !== id)
    setSessions(updated)
    if (activeSessionId === id) {
      if (updated.length > 0) {
        setActiveSessionId(updated[0].id)
      } else {
        setActiveSessionId(null)
      }
    }
  }

  function handleClearAll() {
    if (window.confirm('Are you sure you want to delete all chat history?')) {
      setSessions([])
      setActiveSessionId(null)
      setInput('')
      setSelectedFile(null)
    }
  }

  // Handle starter prompt clicks
  function handleStarterClick(prompt) {
    if (prompt.text) {
      setInput(prompt.text)
    } else {
      // Suggest file upload
      const fileInput = document.getElementById('chat-file-upload')
      if (fileInput) fileInput.click()
    }
  }

  // File drag & drop handlers
  function handleDragOver(e) {
    e.preventDefault()
    setDragOver(true)
  }

  function handleDragLeave(e) {
    e.preventDefault()
    setDragOver(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  async function handleSend() {
    const currentInput = input.trim()
    const currentFile = selectedFile

    if (!currentInput && !currentFile) return
    if (!activeSessionId) return

    setLoading(true)
    
    // Clear attachment controls
    setInput('')
    setSelectedFile(null)

    const isAudio = currentFile?.type?.startsWith('audio/') || currentFile?.name?.endsWith('.wav') || currentFile?.name?.endsWith('.mp3')
    const isVideo = currentFile?.type?.startsWith('video/') || currentFile?.name?.endsWith('.mp4') || currentFile?.name?.endsWith('.avi') || currentFile?.name?.endsWith('.mov')

    // 1. Create user message
    const userMsgId = `msg_${Date.now()}_user`
    const userMessage = {
      id: userMsgId,
      sender: 'user',
      text: currentInput,
      file: currentFile ? { name: currentFile.name, size: currentFile.size } : null,
      isAudio,
      isVideo
    }

    // 2. Create assistant placeholder message
    const assistantMsgId = `msg_${Date.now() + 1}_assistant`
    const assistantMessage = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      isAudio,
      isVideo,
      loading: true,
      result: null,
      error: null
    }

    // Update active session with user & loading message
    let sessionTitle = activeSession ? activeSession.title : 'New Analysis'
    if (sessionTitle === 'New Analysis') {
      if (currentInput) {
        sessionTitle = currentInput.length > 25 ? `${currentInput.substring(0, 25)}...` : currentInput
      } else if (currentFile) {
        sessionTitle = `Verify: ${currentFile.name}`
      }
    }

    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        return {
          ...s,
          title: sessionTitle,
          messages: [...s.messages, userMessage, assistantMessage]
        }
      }
      return s
    }))

    // 3. Make API requests
    try {
      let data
      if (currentFile) {
        if (isAudio) {
          data = await analyzeAudio(currentFile, lang, threshold)
        } else if (isVideo) {
          data = await analyzeVideo(currentFile)
        } else {
          throw new Error('Unsupported file type. Please upload audio or video files only.')
        }
      } else {
        data = await analyzeText(currentInput, threshold)
      }

      // Success: update assistant message in state
      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          const updatedMessages = s.messages.map(m => {
            if (m.id === assistantMsgId) {
              return {
                ...m,
                loading: false,
                result: data
              }
            }
            return m
          })
          return { ...s, messages: updatedMessages }
        }
        return s
      }))

    } catch (err) {
      // Error: update assistant message in state
      const errMsg = err.response?.data?.detail || err.message || 'Verification workflow encountered an error.'
      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          const updatedMessages = s.messages.map(m => {
            if (m.id === assistantMsgId) {
              return {
                ...m,
                loading: false,
                error: errMsg
              }
            }
            return m
          })
          return { ...s, messages: updatedMessages }
        }
        return s
      }))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className={`app-shell ${dragOver ? 'drag-active' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Header
        ollamaAvailable={ollamaAvailable}
        threshold={threshold}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(prev => !prev)}
      />

      <div className="page-grid">
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onClearAll={handleClearAll}
          threshold={threshold}
          onThresholdChange={setThreshold}
          ollamaAvailable={ollamaAvailable}
          sidebarOpen={sidebarOpen}
          onCloseSidebar={() => setSidebarOpen(false)}
        />

        <main className="workspace" id="main-content">
          <div className="chat-container">
            <div className="chat-messages-wrapper">
              <MessageList
                messages={messages}
                onStarterClick={handleStarterClick}
              />
            </div>
            
            <div className="chat-input-wrapper">
              <ChatInput
                input={input}
                setInput={setInput}
                selectedFile={selectedFile}
                setSelectedFile={setSelectedFile}
                lang={lang}
                setLang={setLang}
                onSubmit={handleSend}
                loading={loading}
              />
            </div>
          </div>
        </main>
      </div>

      {dragOver && (
        <div className="drag-overlay">
          <div className="drag-overlay-card">
            <span className="drag-icon" style={{ display: 'inline-flex', justifyContent: 'center', color: 'var(--primary)', marginBottom: '16px' }}>
              <svg viewBox="0 0 24 24" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
              </svg>
            </span>
            <h2>Drop file to attach</h2>
            <p>Upload WAV, MP3 audio or MP4, AVI, MOV video for verification analysis</p>
          </div>
        </div>
      )}
    </div>
  )
}

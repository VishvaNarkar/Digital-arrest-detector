import { useState, useEffect } from 'react'
import Header from './components/Layout/Header'
import Sidebar from './components/Layout/Sidebar'
import TextAnalysis from './components/TextAnalysis'
import AudioAnalysis from './components/AudioAnalysis'
import VideoAnalysis from './components/VideoAnalysis'
import { getHealth } from './api/client'

const TABS = [
  { id: 'text',  icon: '', label: 'Text' },
  { id: 'audio', icon: '', label: 'Audio' },
  { id: 'video', icon: '', label: 'Video' },
]

export default function App() {
  const [activeTab, setActiveTab]           = useState('text')
  const [threshold, setThreshold]           = useState(0.35)
  const [ollamaAvailable, setOllamaAvailable] = useState(false)

  // Poll health on mount and every 30 s
  useEffect(() => {
    async function check() {
      try {
        const h = await getHealth()
        setOllamaAvailable(!!h.ollama_available)
      } catch {
        setOllamaAvailable(false)
      }
    }
    check()
    const id = setInterval(check, 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app-shell">
      <Header ollamaAvailable={ollamaAvailable} />

      <div className="app-body">
        <Sidebar
          threshold={threshold}
          onThresholdChange={setThreshold}
          ollamaAvailable={ollamaAvailable}
        />

        <main className="main-content">
          {/* Tab navigation */}
          <nav className="tabs" role="tablist" aria-label="Analysis channels">
            {TABS.map(tab => (
              <button
                key={tab.id}
                role="tab"
                aria-selected={activeTab === tab.id}
                className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
                id={`tab-${tab.id}`}
              >
                <span className="tab-icon">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Tab panels */}
          {activeTab === 'text'  && <TextAnalysis  threshold={threshold} />}
          {activeTab === 'audio' && <AudioAnalysis threshold={threshold} />}
          {activeTab === 'video' && <VideoAnalysis />}
        </main>
      </div>
    </div>
  )
}

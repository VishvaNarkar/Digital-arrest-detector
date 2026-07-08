import { useState } from 'react'
import { analyzeAudio } from '../api/client'
import ResultCard from './common/ResultCard'

const LANG_OPTIONS = [
  { value: 'en-in', label: 'Indian English (en-in)' },
  { value: 'hi',    label: 'Hindi (hi)' },
  { value: 'gu',    label: 'Gujarati (gu)' },
]

export default function AudioAnalysis({ threshold }) {
  const [file, setFile]         = useState(null)
  const [lang, setLang]         = useState('en-in')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [dragOver, setDragOver] = useState(false)

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
  }

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeAudio(file, lang, threshold)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Audio analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-title">
        <div className="card-title-icon"></div>
        Audio Analysis
      </div>

      {/* Drop zone */}
      <div className="form-group">
        <label className="form-label">Upload audio file</label>
        <div
          className={`dropzone${dragOver ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
        >
          <input
            type="file"
            accept=".wav,.mp3,audio/*"
            onChange={e => handleFile(e.target.files[0])}
            disabled={loading}
            id="audio-file-input"
          />
          <div className="dropzone-icon"></div>
          <div className="dropzone-text">Drag & drop or click to upload</div>
          <div className="dropzone-hint">WAV or MP3 • max 50 MB</div>
          {file && <div className="dropzone-filename">{file.name}</div>}
        </div>
      </div>

      {/* Language */}
      <div className="form-group">
        <label className="form-label" htmlFor="lang-select">Transcription language</label>
        <select
          id="lang-select"
          className="input-field"
          value={lang}
          onChange={e => setLang(e.target.value)}
          disabled={loading}
        >
          {LANG_OPTIONS.map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      <button
        className="btn btn-primary btn-full"
        onClick={handleAnalyze}
        disabled={loading || !file}
        id="analyze-audio-btn"
      >
        {loading ? (
          <>
            <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
            Transcribing &amp; analyzing…
          </>
        ) : 'Analyze Audio'}
      </button>

      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <div className="spinner-text">Running Vosk transcription — this may take a moment…</div>
        </div>
      )}

      {error && (
        <div className="alert danger" style={{ marginTop: 16 }}>{error}</div>
      )}

      {result && !loading && (
        <div className="result-section">
          <ResultCard result={result.analysis} transcription={result.transcription} />
        </div>
      )}
    </div>
  )
}

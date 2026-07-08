import { useState } from 'react'
import { analyzeText } from '../api/client'
import ResultCard from './common/ResultCard'

export default function TextAnalysis({ threshold }) {
  const [text, setText]     = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function handleAnalyze() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeText(text.trim(), threshold)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-title">
        <div className="card-title-icon"></div>
        Text Analysis
      </div>

      <div className="form-group">
        <label className="form-label" htmlFor="text-input">
          Message, SMS, or email text
        </label>
        <textarea
          id="text-input"
          className="input-field"
          rows={6}
          placeholder="Paste the suspicious message here… (English, Hindi, or Gujarati)"
          value={text}
          onChange={e => setText(e.target.value)}
          disabled={loading}
        />
      </div>

      <button
        className="btn btn-primary btn-full"
        onClick={handleAnalyze}
        disabled={loading || !text.trim()}
        id="analyze-text-btn"
      >
        {loading ? (
          <>
            <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
            Analyzing…
          </>
        ) : 'Analyze Text'}
      </button>

      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <div className="spinner-text">Running hybrid detection…</div>
        </div>
      )}

      {error && (
        <div className="alert danger" style={{ marginTop: 16 }}>
          {error}
        </div>
      )}

      {result && !loading && (
        <div className="result-section">
          <ResultCard result={result} />
        </div>
      )}
    </div>
  )
}

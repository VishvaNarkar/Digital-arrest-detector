import { useState } from 'react'
import { analyzeVideo } from '../api/client'
import ResultCard from './common/ResultCard'

export default function VideoAnalysis() {
  const [file, setFile]         = useState(null)
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
      const data = await analyzeVideo(file)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Video analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-title">
        <div className="card-title-icon"></div>
        Video Deepfake Detection
      </div>

      <div className="form-group">
        <label className="form-label">Upload video file</label>
        <div
          className={`dropzone${dragOver ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
        >
          <input
            type="file"
            accept=".mp4,.avi,.mov,video/*"
            onChange={e => handleFile(e.target.files[0])}
            disabled={loading}
            id="video-file-input"
          />
          <div className="dropzone-icon"></div>
          <div className="dropzone-text">Drag & drop or click to upload</div>
          <div className="dropzone-hint">MP4 · AVI · MOV • max 200 MB</div>
          {file && <div className="dropzone-filename">{file.name}</div>}
        </div>
      </div>

      <div className="alert info" style={{ marginBottom: 16, fontSize: '0.8rem' }}>
        ℹ️ The detector samples 12 evenly-spaced frames and averages their deepfake probability scores.
      </div>

      <button
        className="btn btn-primary btn-full"
        onClick={handleAnalyze}
        disabled={loading || !file}
        id="analyze-video-btn"
      >
        {loading ? (
          <>
            <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
            Analyzing frames…
          </>
        ) : 'Analyze Video'}
      </button>

      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <div className="spinner-text">Sampling frames &amp; running deepfake model…</div>
        </div>
      )}

      {error && (
        <div className="alert danger" style={{ marginTop: 16 }}>{error}</div>
      )}

      {result && !loading && (
        <div className="result-section">
          <ResultCard result={result} />
          {result.label?.includes('Deepfake') ? (
            <div className="alert danger" style={{ marginTop: 14 }}>
              This video shows signs of AI manipulation. Verify the source before sharing or acting on it.
            </div>
          ) : (
            <div className="alert success" style={{ marginTop: 14 }}>
              No significant deepfake indicators detected. Always verify high-stakes video independently.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

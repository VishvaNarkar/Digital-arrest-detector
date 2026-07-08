import CircularProgress from './CircularProgress'
import KeywordBadges from './KeywordBadges'
import RagSection from './RagSection'

function riskColor(pct) {
  if (pct < 40) return '#16A34A'
  if (pct < 70) return '#D97706'
  return '#DC2626'
}

/**
 * Universal result card used by all three analysis tabs.
 * @param {Object} result  — the analysis object from the API
 * @param {string} [transcription] — optional transcription text (audio tab)
 */
export default function ResultCard({ result, transcription }) {
  const isScam   = result.label?.includes('Scam') || result.label?.includes('Deepfake')
  const variant  = isScam ? 'scam' : 'safe'
  const pct      = Math.round((result.combined_prob ?? result.score ?? 0) * 100)
  const color    = riskColor(pct)

  return (
    <div className={`result-card ${variant}`}>
      {/* Header row */}
      <div className="result-header">
        <div className="result-icon-wrap">{isScam ? '' : ''}</div>
        <div className="result-label-group">
          <div className={`result-verdict ${variant}`}>{result.label}</div>
          {result.category && (
            <div className="result-category">{result.category}</div>
          )}
        </div>
        <CircularProgress percent={pct} color={color} />
      </div>

      {/* Transcription (audio tab only) */}
      {transcription && (
        <div style={{ marginBottom: 16 }}>
          <div className="keywords-title" style={{ marginBottom: 6 }}>Transcription</div>
          <div className="transcription-box">{transcription}</div>
        </div>
      )}

      {/* Video progress bar */}
      {result.percent !== undefined && result.score !== undefined && (
        <div className="video-bar-wrap">
          <div className="video-bar-label">
            <span>Deepfake probability</span>
            <span>{result.percent}%</span>
          </div>
          <div className="video-bar-track">
            <div
              className="video-bar-fill"
              style={{ width: `${result.percent}%`, background: color }}
            />
          </div>
        </div>
      )}

      {/* Meta grid (text/audio only) */}
      {result.ml_prob !== undefined && (
        <div className="meta-grid">
          <div className="meta-item">
            <div className="meta-item-label">ML Prob</div>
            <div className="meta-item-value">{(result.ml_prob * 100).toFixed(1)}%</div>
          </div>
          <div className="meta-item">
            <div className="meta-item-label">Keywords</div>
            <div className="meta-item-value">{result.keyword_score ?? 0}</div>
          </div>
          <div className="meta-item">
            <div className="meta-item-label">Sentiment</div>
            <div className="meta-item-value">
              {(result.sentiment?.compound ?? 0).toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Advice banner */}
      {result.ml_prob !== undefined && (
        <div className={`advice-banner ${variant}`}>
          <span>{isScam ? '' : ''}</span>
          <span>
            {isScam
              ? 'Do not click suspicious links or share OTPs. Verify the sender through official channels.'
              : 'Content appears safe. Always double-check unknown callers, senders, or links.'}
          </span>
        </div>
      )}

      {/* Keywords */}
      {result.keywords?.length > 0 && (
        <KeywordBadges keywords={result.keywords} />
      )}

      {/* RAG reasoning */}
      {'rag' in result && (
        <RagSection rag={result.rag} />
      )}
    </div>
  )
}

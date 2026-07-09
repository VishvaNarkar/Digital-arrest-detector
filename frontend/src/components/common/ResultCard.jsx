import CircularProgress from './CircularProgress'
import KeywordBadges from './KeywordBadges'
import RagSection from './RagSection'

function riskColor(percent) {
  if (percent < 40) return '#22c55e'
  if (percent < 70) return '#f59e0b'
  return '#ef4444'
}

export default function ResultCard({ result, transcription }) {
  const isRisky = result.label?.includes('Scam') || result.label?.includes('Deepfake')
  const variant = isRisky ? 'risk' : 'safe'
  const percent = Math.round((result.combined_prob ?? result.score ?? 0) * 100)
  const color = riskColor(percent)
  const metrics = [
    { label: 'Model score', value: `${((result.ml_prob ?? result.score ?? 0) * 100).toFixed(1)}` },
    { label: 'Keyword score', value: `${result.keyword_score ?? 0}` },
    { label: 'Sentiment', value: `${(result.sentiment?.compound ?? 0).toFixed(2)}` },
  ]

  return (
    <article className={`result-card ${variant}`}>
      <div className="result-card-header">
        <div>
          <div className="result-kicker">Result</div>
          <h4>{result.label}</h4>
          {result.category && <p className="result-subtitle">{result.category}</p>}
        </div>
        <CircularProgress percent={percent} color={color} />
      </div>

      <div className="result-tags">
        <span className={`result-tag ${variant}`}>{isRisky ? 'Review carefully' : 'Low concern'}</span>
        <span className="result-tag neutral">{percent}% overall risk</span>
      </div>

      <section className="metric-grid">
        {metrics.map(metric => (
          <div className="metric-card" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}{metric.label === 'Model score' ? '%' : ''}</strong>
          </div>
        ))}
      </section>

      {transcription && (
        <section className="stack-section">
          <div className="section-label">Transcription</div>
          <div className="transcription-box">{transcription}</div>
        </section>
      )}

      {result.percent !== undefined && result.score !== undefined && (
        <section className="stack-section">
          <div className="bar-row">
            <span>Deepfake probability</span>
            <strong>{result.percent}%</strong>
          </div>
          <div className="video-bar-track">
            <div className="video-bar-fill" style={{ width: `${result.percent}%`, background: color }} />
          </div>
        </section>
      )}

      <div className={`advice-banner ${variant}`}>
        <span>{isRisky ? 'Action' : 'Note'}</span>
        <span>
          {isRisky
            ? 'Verify the source through official channels before taking any action.'
            : 'Content appears lower risk, but high-stakes decisions should still be verified independently.'}
        </span>
      </div>

      {result.keywords?.length > 0 && <KeywordBadges keywords={result.keywords} />}
      {'rag' in result && <RagSection rag={result.rag} />}
    </article>
  )
}

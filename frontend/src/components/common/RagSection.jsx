import { useState } from 'react'

export default function RagSection({ rag }) {
  const [open, setOpen] = useState(false)

  if (!rag) {
    return (
      <div className="alert info" style={{ marginTop: 14 }}>
        LLM reasoning unavailable — Ollama is offline or skipped. Showing heuristic/ML results only.
      </div>
    )
  }

  const explanation = rag.explanation || (typeof rag.raw === 'string' ? rag.raw : '')
  const advice      = Array.isArray(rag.advice) ? rag.advice : []
  const rp          = rag.risk_percent ?? rag.llm_prob
  let rpPct = null
  if (rp != null) {
    const val = parseFloat(rp)
    rpPct = val <= 1 ? Math.round(val * 100) : Math.round(val)
  }

  return (
    <div className="rag-section" style={{ marginTop: 16 }}>
      <button className="rag-toggle" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span>AI Reasoning — Phi3:mini{rpPct != null ? ` · ${rpPct}% risk` : ''}</span>
        <i className={`rag-toggle-icon${open ? ' open' : ''}`}>▾</i>
      </button>

      {open && (
        <div className="rag-body">
          {explanation && (
            <>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
                Explanation
              </div>
              <div className="rag-explanation">{explanation}</div>
            </>
          )}

          {advice.length > 0 && (
            <>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
                Advice
              </div>
              <ul className="rag-advice-list">
                {advice.map((a, i) => (
                  <li key={i} className="rag-advice-item">{a}</li>
                ))}
              </ul>
            </>
          )}

          {rpPct != null && (
            <div className="rag-risk-row">
              LLM risk estimate: <span className="rag-risk-pill">{rpPct}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

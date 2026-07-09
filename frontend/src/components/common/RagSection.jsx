import { useState } from 'react'

export default function RagSection({ rag }) {
  const [open, setOpen] = useState(false)

  if (!rag) {
    return <div className="alert info spaced">LLM reasoning unavailable. Showing heuristic and ML results only.</div>
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
    <div className="rag-section">
      <button className="rag-toggle" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span>AI reasoning{rpPct != null ? ` - ${rpPct}% risk` : ''}</span>
        <i className={`rag-toggle-icon${open ? ' open' : ''}`}>v</i>
      </button>

      {open && (
        <div className="rag-body">
          {explanation && (
            <section className="stack-section">
              <div className="section-label">Explanation</div>
              <div className="rag-explanation">{explanation}</div>
            </section>
          )}

          {advice.length > 0 && (
            <section className="stack-section">
              <div className="section-label">Advice</div>
              <ul className="rag-advice-list">
                {advice.map((a, i) => (
                  <li key={i} className="rag-advice-item">{a}</li>
                ))}
              </ul>
            </section>
          )}

          {rpPct != null && (
            <div className="rag-risk-row">
              LLM risk estimate <span className="rag-risk-pill">{rpPct}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

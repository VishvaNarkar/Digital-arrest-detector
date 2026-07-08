import { useEffect, useRef } from 'react'

/**
 * Animated SVG circular progress ring.
 * @param {number} percent  0-100
 * @param {string} color    stroke color
 */
export default function CircularProgress({ percent, color }) {
  const RADIUS = 30
  const STROKE = 6
  const SIZE = (RADIUS + STROKE) * 2
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS
  const pct = Math.max(0, Math.min(100, Math.round(percent)))
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE

  return (
    <div className="circular-progress" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        <circle
          className="track"
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          strokeWidth={STROKE}
        />
        <circle
          className="fill"
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          strokeWidth={STROKE}
          stroke={color}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
        />
      </svg>
      <div className="pct-text">{pct}%</div>
    </div>
  )
}

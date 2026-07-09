export default function CircularProgress({ percent, color }) {
  const radius = 30
  const stroke = 7
  const size = (radius + stroke) * 2
  const circumference = 2 * Math.PI * radius
  const pct = Math.max(0, Math.min(100, Math.round(percent)))
  const offset = circumference - (pct / 100) * circumference

  return (
    <div className="circular-progress" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          className="track"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={stroke}
        />
        <circle
          className="fill"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={stroke}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
        />
      </svg>
      <div className="pct-text">{pct}%</div>
    </div>
  )
}

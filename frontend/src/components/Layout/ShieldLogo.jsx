export default function ShieldLogo({ size = 36 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#4F6EF7" />
          <stop offset="100%" stopColor="#7C3AED" />
        </linearGradient>
      </defs>
      <path
        d="M20 3L5 9v11c0 9.39 6.42 18.18 15 20.38C28.58 38.18 35 29.39 35 20V9L20 3z"
        fill="url(#sg)"
      />
      <path
        d="M14 20l4.5 4.5L27 15"
        stroke="white"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

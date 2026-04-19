/* TrustGauge.jsx — Clean strokeDasharray-based semicircle gauge */

const TIER_COLOR = (score) => {
  if (score >= 80) return '#22c55e';
  if (score >= 60) return '#84cc16';
  if (score >= 40) return '#eab308';
  if (score >= 20) return '#f97316';
  return '#ef4444';
};

export default function TrustGauge({ score }) {
  const color = TIER_COLOR(score);

  const r          = 72;
  const cx         = 100;
  const cy         = 95;
  const strokeW    = 11;
  const halfCirc   = Math.PI * r;          // arc length of one semicircle ≈ 226
  const safe       = Math.max(0, Math.min(100, score));
  const dashOffset = halfCirc * (1 - safe / 100);  // 0 = full, halfCirc = empty

  const ticks = [0, 25, 50, 75, 100];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg
        width="200"
        height="108"
        viewBox="0 0 200 108"
        style={{ display: 'block', shapeRendering: 'geometricPrecision' }}
      >
        {/* ── Background track (static, full semicircle) ───────────── */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={strokeW}
          strokeLinecap="round"
          strokeDasharray={`${halfCirc} ${halfCirc * 10}`}
          transform={`rotate(180 ${cx} ${cy})`}
        />

        {/* ── Coloured progress arc ─────────────────────────────────── */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeW}
          strokeLinecap="round"
          strokeDasharray={`${halfCirc} ${halfCirc * 10}`}
          strokeDashoffset={dashOffset}
          transform={`rotate(180 ${cx} ${cy})`}
          style={{
            transition: 'stroke-dashoffset 0.9s cubic-bezier(0.4,0,0.2,1), stroke 0.4s ease',
          }}
        />

        {/* ── Tick labels ───────────────────────────────────────────── */}
        {ticks.map((v) => {
          // angle goes from π (left, v=0) to 0 (right, v=100)
          const a  = Math.PI * (1 - v / 100);
          const tx = cx + (r + 15) * Math.cos(a);
          const ty = cy - (r + 15) * Math.sin(a);
          return (
            <text
              key={v}
              x={tx}
              y={ty}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="8.5"
              fontFamily="Inter, sans-serif"
              fill="rgba(255,255,255,0.28)"
            >
              {v}
            </text>
          );
        })}
      </svg>

      {/* Numeric score below the gauge */}
      <div style={{
        fontSize: '2.8rem',
        fontWeight: 800,
        color,
        letterSpacing: '-0.03em',
        lineHeight: 1,
        marginTop: -4,
      }}>
        {score}
      </div>
      <div style={{ fontSize: '.72rem', color: 'rgba(255,255,255,.32)', marginTop: 3 }}>
        / 100
      </div>
    </div>
  );
}

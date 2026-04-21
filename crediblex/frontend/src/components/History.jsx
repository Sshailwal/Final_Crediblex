/* History.jsx — Shows last 5 analyses, saved in localStorage */
const TIER_COLOR = (score) => {
  if (score >= 80) return '#22c55e';
  if (score >= 60) return '#84cc16';
  if (score >= 40) return '#eab308';
  if (score >= 20) return '#f97316';
  return '#ef4444';
};

const TIER_LABEL = (score) => {
  if (score >= 80) return 'High Trust';
  if (score >= 60) return 'Mostly Credible';
  if (score >= 40) return 'Mixed';
  if (score >= 20) return 'Low Trust';
  return 'Unreliable';
};

export default function History({ history, onSelect, onClear }) {
  if (!history || history.length === 0) return null;

  return (
    <div style={{
      maxWidth: 720,
      margin: '0 auto 24px auto',
      padding: '0 16px',
    }}>

      {/* ── Header ── */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
      }}>
        <span style={{
          fontSize: '0.75rem',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'var(--text-muted, #888)',
        }}>
          🕘 Recent Analyses
        </span>
        <button
          onClick={onClear}
          style={{
            background: 'none',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 6,
            color: 'var(--text-muted, #888)',
            fontSize: '0.72rem',
            padding: '3px 10px',
            cursor: 'pointer',
          }}
        >
          Clear
        </button>
      </div>

      {/* ── History Cards ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {history.map((item, i) => {
          const color = TIER_COLOR(item.score);
          const label = TIER_LABEL(item.score);
          return (
            <div
              key={i}
              onClick={() => onSelect(item)}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 10,
                padding: '10px 14px',
                cursor: 'pointer',
                transition: 'background 0.2s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
              onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
            >
              {/* Left — title and time */}
              <div style={{ overflow: 'hidden', marginRight: 12 }}>
                <div style={{
                  fontSize: '0.82rem',
                  color: 'var(--text, #eee)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  maxWidth: 420,
                }}>
                  {item.type === 'text' ? '💬 ' : '🔗 '}
                  {item.title || item.value?.slice(0, 80) || 'Unknown'}
                </div>
                <div style={{ fontSize: '0.70rem', color: 'var(--text-muted, #888)', marginTop: 2 }}>
                  {item.time}
                </div>
              </div>

              {/* Right — score badge */}
              <div style={{
                flexShrink: 0,
                textAlign: 'center',
                background: `${color}18`,
                border: `1px solid ${color}44`,
                borderRadius: 8,
                padding: '4px 10px',
              }}>
                <div style={{ fontSize: '1rem', fontWeight: 700, color }}>{item.score}</div>
                <div style={{ fontSize: '0.62rem', color, opacity: 0.85 }}>{label}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
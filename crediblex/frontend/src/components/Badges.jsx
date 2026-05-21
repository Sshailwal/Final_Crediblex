/* Badges.jsx — Factuality bar, Intent chip, Emotion chip */

function FactualityBar({ value }) {
  const pct = Math.max(0, Math.min(100, Math.round((Number(value) || 0) * 100)));
  const color =
    pct >= 70 ? '#22c55e' :
    pct >= 50 ? '#eab308' : '#ef4444';

  return (
    <div>
      <div className="fact-bar-bg">
        <div
          className="fact-bar-fill"
          style={{ width:`${pct}%`, background:`linear-gradient(90deg, ${color}99, ${color})` }}
        />
      </div>
    </div>
  );
}

function IntentChip({ intent }) {
  const normalized = intent || 'Unknown';
  const cls = {
    News:    'chip-news',
    Opinion: 'chip-opinion',
    Satire:  'chip-satire',
  }[normalized] || 'chip-emotion';

  const icon = { News: '📰', Opinion: '💬', Satire: '🎭' }[normalized] || '🔍';

  return (
    <div>
      <span className={`chip ${cls}`}>{icon} {normalized}</span>
    </div>
  );
}

function EmotionChip({ emotion }) {
  const normalized = String(emotion || 'Neutral');
  const key = normalized.toLowerCase();
  const icons = {
    anger:'😡', fear:'😨', disgust:'🤢', sadness:'😢', neutral:'😐',
    joy:'😄', love:'❤️', admiration:'🤩', gratitude:'🙏', optimism:'✨',
    curiosity:'🤔', surprise:'😮', caring:'💙', approval:'👍', excitement:'🎉',
  };
  const icon = icons[key] || '🎭';

  return (
    <div>
      <span className="chip chip-emotion" style={{ textTransform: 'capitalize' }}>{icon} {normalized}</span>
    </div>
  );
}

export { FactualityBar, IntentChip, EmotionChip };

/* Badges.jsx — Factuality bar, Intent chip, Emotion chip */

function FactualityBar({ value }) {
  const pct = Math.round(value * 100);
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
  const cls = {
    News:    'chip-news',
    Opinion: 'chip-opinion',
    Satire:  'chip-satire',
  }[intent] || 'chip-emotion';

  const icon = { News: '📰', Opinion: '💬', Satire: '🎭' }[intent] || '🔍';

  return (
    <div>
      <span className={`chip ${cls}`}>{icon} {intent}</span>
    </div>
  );
}

function EmotionChip({ emotion }) {
  const icons = {
    anger:'😡', fear:'😨', disgust:'🤢', sadness:'😢', neutral:'😐',
    joy:'😄', love:'❤️', admiration:'🤩', gratitude:'🙏', optimism:'✨',
    curiosity:'🤔', surprise:'😮', caring:'💙', approval:'👍', excitement:'🎉',
  };
  const icon = icons[emotion] || '🎭';

  return (
    <div>
      <span className="chip chip-emotion" style={{ textTransform: 'capitalize' }}>{icon} {emotion}</span>
    </div>
  );
}

export { FactualityBar, IntentChip, EmotionChip };

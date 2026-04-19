/* UrlInput.jsx — Dual-mode input: News URL or WhatsApp / raw text */
import { useState } from 'react';

const SpinIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
    style={{ animation: 'spin .8s linear infinite' }}>
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
  </svg>
);

const SearchIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
  </svg>
);

export default function UrlInput({ onSubmit, loading }) {
  const [mode, setMode]   = useState('url');   // 'url' | 'text'
  const [url, setUrl]     = useState('');
  const [text, setText]   = useState('');

  const charCount = text.length;
  const charOk    = charCount >= 50;
  const isValid   = mode === 'url' ? url.trim().length > 0 : charOk;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid || loading) return;
    if (mode === 'url') {
      onSubmit({ type: 'url', value: url.trim() });
    } else {
      onSubmit({ type: 'text', value: text.trim() });
    }
  };

  return (
    <div className="input-card">

      {/* ── Mode Tabs ────────────────────────────────────────────────────── */}
      <div className="input-tabs">
        <button
          id="tab-url"
          className={`input-tab ${mode === 'url' ? 'active' : ''}`}
          onClick={() => setMode('url')}
          type="button"
          disabled={loading}
        >
          🔗 News URL
        </button>
        <button
          id="tab-text"
          className={`input-tab ${mode === 'text' ? 'active' : ''}`}
          onClick={() => setMode('text')}
          type="button"
          disabled={loading}
        >
          💬 WhatsApp / Text
        </button>
      </div>

      {/* ── Form ─────────────────────────────────────────────────────────── */}
      <form onSubmit={handleSubmit}>

        {mode === 'url' ? (
          /* URL mode */
          <div className="input-row">
            <input
              id="url-input"
              className="url-input"
              type="text"
              placeholder="https://www.bbc.com/news/..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={loading}
              spellCheck={false}
              autoComplete="off"
            />
            <button id="analyze-url-btn" className="analyze-btn" type="submit"
              disabled={loading || !url.trim()}>
              {loading ? <><SpinIcon /> Analyzing…</> : <><SearchIcon /> Analyze</>}
            </button>
          </div>
        ) : (
          /* WhatsApp / Text mode */
          <div>
            <textarea
              id="text-input"
              className="text-input"
              placeholder="Paste a WhatsApp forward, news message, or any article text here to fact-check it…

Example: 'Breaking news — the government has secretly imposed emergency rule! Forward to everyone NOW!'

Minimum 50 characters."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading}
              rows={7}
            />
            <div className="text-input-footer">
              <span className={`char-count ${charOk ? 'ok' : 'warn'}`}>
                {charCount} chars {charOk ? '✓ ready' : `(need ${50 - charCount} more)`}
              </span>
              <button id="factcheck-btn" className="analyze-btn factcheck-btn" type="submit"
                disabled={loading || !isValid}>
                {loading ? <><SpinIcon /> Fact-checking…</> : <><CheckIcon /> Fact Check</>}
              </button>
            </div>
          </div>
        )}

        <p className="input-hint">
          {mode === 'url'
            ? 'Paste any public news article URL — the model will scrape & score it in seconds.'
            : 'Paste raw text from WhatsApp, Telegram, or any source. No URL needed.'}
        </p>
      </form>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

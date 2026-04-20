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

// ── URL Validation ────────────────────────────────────────────────────────────
const validateUrl = (url) => {
  if (!url.trim()) return { valid: false, message: '' };

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return { valid: false, message: '⚠️ URL must start with http:// or https://' };
  }

  const urlPattern = /^https?:\/\/([\w-]+(\.[\w-]+)+)([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?$/;
  if (!urlPattern.test(url.trim())) {
    return { valid: false, message: '⚠️ Please enter a valid URL  e.g. https://www.bbc.com/news/...' };
  }

  return { valid: true, message: '✓ Looks good!' };
};

export default function UrlInput({ onSubmit, loading }) {
  const [mode, setMode]   = useState('url');   // 'url' | 'text'
  const [url, setUrl]     = useState('');
  const [text, setText]   = useState('');
  const [urlTouched, setUrlTouched] = useState(false);  // only show error after user types

  const charCount = text.length;
  const charOk    = charCount >= 50;

  const urlValidation = validateUrl(url);
  const isValidUrl    = urlValidation.valid;
  const isValid       = mode === 'url' ? isValidUrl : charOk;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid || loading) return;
    if (mode === 'url') {
      onSubmit({ type: 'url', value: url.trim() });
    } else {
      onSubmit({ type: 'text', value: text.trim() });
    }
  };

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
    setUrlTouched(true);
  };

  const handleModeSwitch = (newMode) => {
    setMode(newMode);
    setUrlTouched(false);
  };

  // Decide what message to show under the URL input
  const showError   = urlTouched && url.trim() && !isValidUrl;
  const showSuccess = urlTouched && isValidUrl;

  return (
    <div className="input-card">

      {/* ── Mode Tabs ────────────────────────────────────────────────────── */}
      <div className="input-tabs">
        <button
          id="tab-url"
          className={`input-tab ${mode === 'url' ? 'active' : ''}`}
          onClick={() => handleModeSwitch('url')}
          type="button"
          disabled={loading}
        >
          🔗 News URL
        </button>
        <button
          id="tab-text"
          className={`input-tab ${mode === 'text' ? 'active' : ''}`}
          onClick={() => handleModeSwitch('text')}
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
          <div>
            <div className="input-row">
              <input
                id="url-input"
                className="url-input"
                type="text"
                placeholder="https://www.bbc.com/news/..."
                value={url}
                onChange={handleUrlChange}
                disabled={loading}
                spellCheck={false}
                autoComplete="off"
                style={{
                  borderColor: showError ? '#ef4444' : showSuccess ? '#22c55e' : undefined,
                }}
              />
              <button id="analyze-url-btn" className="analyze-btn" type="submit"
                disabled={loading || !isValidUrl}>
                {loading ? <><SpinIcon /> Analyzing…</> : <><SearchIcon /> Analyze</>}
              </button>
            </div>

            {/* ── Validation message ── */}
            {urlTouched && url.trim() && (
              <p style={{
                marginTop: 6,
                fontSize: '0.78rem',
                color: showError ? '#ef4444' : '#22c55e',
                minHeight: '1.2em',
              }}>
                {urlValidation.message}
              </p>
            )}
          </div>
        ) : (
          /* WhatsApp / Text mode */
          <div>
            <textarea
              id="text-input"
              className="text-input"
              placeholder={`Paste a WhatsApp forward, news message, or any article text here to fact-check it…\n\nExample: 'Breaking news — the government has secretly imposed emergency rule! Forward to everyone NOW!'\n\nMinimum 50 characters.`}
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
import { useState, useEffect } from 'react';
import './index.css';
import UrlInput from './components/UrlInput';
import TrustGauge from './components/TrustGauge';
import BiasSlider from './components/BiasSlider';
import { FactualityBar, IntentChip, EmotionChip } from './components/Badges';
import History from './components/History';

const API = 'http://localhost:7860';
const MAX_HISTORY = 5;

const TIER_COLOR = (score) => {
  if (score >= 80) return '#22c55e';
  if (score >= 60) return '#84cc16';
  if (score >= 40) return '#eab308';
  if (score >= 20) return '#f77417';
  return '#ef4444';
};

// ── Copy Report Button ────────────────────────────────────────────────────────
function CopyReportButton({ report, inputType }) {
  const [copied, setCopied] = useState(false);

  const buildText = () => {
    const dim = report?.dimensions || {};
    const findings = (report.key_findings || []).map(f => `  • ${f.text}`).join('\n');
    const source = inputType === 'text' ? '[Pasted Text]' : (report.metadata?.title || 'Unknown');

    return [
      '📊 CredibleX Analysis Report',
      '─'.repeat(34),
      `📰 Title   : ${source}`,
      `⭐ Score   : ${report.score}/100`,
      `✅ Verdict : ${report.verdict}`,
      '',
      '📊 Dimensions:',
      `  🧾 Factuality : ${Math.round((dim.factuality?.value ?? 0) * 100)}%`,
      `  ⚖️  Bias       : ${dim.bias?.value ?? 'N/A'}`,
      `  🎯 Intent     : ${dim.intent?.value ?? 'N/A'}`,
      `  😤 Emotion    : ${dim.emotion?.value ?? 'N/A'}`,
      '',
      findings ? `📋 Key Findings:\n${findings}` : '',
      '',
      `🤖 Summary:\n${report.summary || 'N/A'}`,
      '',
      '─'.repeat(34),
      `Analyzed by CredibleX • ${new Date().toLocaleString()}`,
    ].filter(line => line !== null).join('\n');
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(buildText());
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      alert('Could not copy — please copy manually.');
    }
  };

  return (
    <button
      onClick={handleCopy}
      style={{
        display:     'flex',
        alignItems:  'center',
        gap:         6,
        margin:      '0 auto 20px auto',
        padding:     '9px 20px',
        background:  copied ? 'rgba(34,197,94,0.12)' : 'rgba(108,99,255,0.12)',
        border:      `1px solid ${copied ? 'rgba(34,197,94,0.35)' : 'rgba(108,99,255,0.35)'}`,
        borderRadius: 10,
        color:       copied ? '#4ade80' : '#a78bfa',
        fontSize:    '0.82rem',
        fontWeight:  600,
        cursor:      'pointer',
        transition:  'all 0.2s',
      }}
    >
      {copied ? '✅ Copied!' : '📋 Copy Report'}
    </button>
  );
}

// ── Key Findings bullet list ──────────────────────────────────────────────────
function KeyFindings({ findings }) {
  if (!findings || findings.length === 0) return null;

  const colorMap = {
    good: { bg: 'rgba(34,197,94,.08)',   border: 'rgba(34,197,94,.25)',   text: '#4ade80' },
    warn: { bg: 'rgba(234,179,8,.08)',   border: 'rgba(234,179,8,.25)',   text: '#fde047' },
    bad:  { bg: 'rgba(239,68,68,.08)',   border: 'rgba(239,68,68,.25)',   text: '#f87171' },
    info: { bg: 'rgba(108,99,255,.08)',  border: 'rgba(108,99,255,.25)', text: '#a78bfa' },
  };

  return (
    <div className="key-findings-card">
      <div className="findings-header">📋 Key Findings</div>
      <ul className="findings-list">
        {findings.map((f, i) => {
          const c = colorMap[f.type] || colorMap.info;
          return (
            <li key={i} className="finding-item"
              style={{ background: c.bg, borderLeft: `3px solid ${c.border}` }}>
              <span className="finding-icon">{f.icon}</span>
              <span className="finding-text" style={{ color: c.text }}>{f.text}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ── WhatsApp message metadata card ───────────────────────────────────────────
function TextMetaCard({ meta }) {
  if (!meta) return null;
  return (
    <div className="text-meta-card">
      <div className="findings-header">📱 Message Details</div>
      <div className="text-meta-grid">
        <div className="text-meta-item">
          <span className="text-meta-label">Word Count</span>
          <span className="text-meta-value">{meta.word_count?.toLocaleString()}</span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Reading Time</span>
          <span className="text-meta-value">{meta.estimated_read_time}</span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Contains URL</span>
          <span className="text-meta-value" style={{ color: meta.contains_url ? '#fde047' : '#4ade80' }}>
            {meta.contains_url ? '⚠️ Yes' : '✅ No'}
          </span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Looks Like Forward</span>
          <span className="text-meta-value" style={{ color: meta.looks_like_forward ? '#f87171' : '#4ade80' }}>
            {meta.looks_like_forward ? '🚨 Yes — be cautious' : '✅ No'}
          </span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Numeric Claims</span>
          <span className="text-meta-value">{meta.numeric_claims_count}</span>
        </div>
        {meta.likely_non_english && (
          <div className="text-meta-item" style={{ gridColumn: '1 / -1' }}>
            <span className="text-meta-label">Language</span>
            <span className="text-meta-value" style={{ color: '#fde047' }}>
              ⚠️ Possibly non-English — model accuracy may be reduced
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [state, setState]         = useState('idle');
  const [report, setReport]       = useState(null);
  const [errMsg, setErrMsg]       = useState('');
  const [inputType, setInputType] = useState('url');
  const [history, setHistory]     = useState([]);

  // ── Load history from localStorage on startup ──
  useEffect(() => {
    try {
      const saved = localStorage.getItem('crediblex_history');
      if (saved) setHistory(JSON.parse(saved));
    } catch {
      setHistory([]);
    }
  }, []);

  // ── Save a new result to history ──
  const saveToHistory = (report, type, value) => {
    try {
      const entry = {
        type,
        value,
        title: report.metadata?.title || (type === 'text' ? value.slice(0, 60) : value) || 'Analysis',
        score: report.score ?? 0,
        verdict: report.verdict || 'Unknown',
        report,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setHistory(prev => {
        const updated = [entry, ...prev].slice(0, MAX_HISTORY);
        try {
          localStorage.setItem('crediblex_history', JSON.stringify(updated));
        } catch (e) {
          console.error('History save failed:', e);
        }
        return updated;
      });
    } catch (err) {
      console.error('Error preparing history entry:', err);
    }
  };

  // ── Clear history ──
  const clearHistory = () => {
    localStorage.removeItem('crediblex_history');
    setHistory([]);
  };

  // ── Click a history item to restore that report ──
  const handleHistorySelect = (item) => {
    setReport(item.report);
    setInputType(item.type);
    setState('success');
  };

  const analyze = async ({ type, value }) => {
    setState('loading');
    setReport(null);
    setErrMsg('');
    setInputType(type);

    try {
      const endpoint = type === 'url' ? `${API}/analyze-url` : `${API}/analyze-text`;
      const body     = type === 'url' ? { url: value } : { text: value };

      const res  = await fetch(endpoint, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        const msg = data?.detail?.message || data?.detail || 'Unknown error from server.';
        throw new Error(msg);
      }
      setReport(data);
      setState('success');
      saveToHistory(data, type, value);
    } catch (err) {
      setErrMsg(err.message || 'Network error — is the API running on port 8000?');
      setState('error');
    }
  };

  const tierColor = report ? TIER_COLOR(report.score) : '#6c63ff';
  const dim       = report?.dimensions || {};
  const isText    = inputType === 'text';

  return (
    <div className="app">

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <header className="hero">
        <div className="hero-logo">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2L3 7l9 5 9-5-9-5zM3 12l9 5 9-5M3 17l9 5 9-5"/>
          </svg>
          CredibleX
        </div>
        <h1>News Trust Score</h1>
        <p>AI-powered credibility analysis — factuality, bias, intent &amp; emotion in seconds.</p>
      </header>

      {/* ── Input ─────────────────────────────────────────────────────────── */}
      <UrlInput onSubmit={analyze} loading={state === 'loading'} />

      {/* ── History Panel ─────────────────────────────────────────────────── */}
      <History
        history={history}
        onSelect={handleHistorySelect}
        onClear={clearHistory}
      />

      {/* ── Status ────────────────────────────────────────────────────────── */}
      <div className="status-area">
        {state === 'loading' && (
          <>
            <div className="loading-bar" />
            <p className="loading-text">
              {isText
                ? 'Fact-checking message — running model inference…'
                : 'Scraping article & running model inference…'}
            </p>
          </>
        )}
        {state === 'error' && (
          <div className="error-box">
            <strong>Analysis Failed</strong>{errMsg}
          </div>
        )}
      </div>

      {/* ── Report ────────────────────────────────────────────────────────── */}
      {state === 'success' && report && (
        <div className="report">

          {/* Metadata */}
          <div className="report-meta">
            {isText && (
              <div className="whatsapp-badge">💬 WhatsApp / Text Analysis</div>
            )}
            <h2>{report.metadata?.title || (isText ? 'Message Analysis' : 'Article')}</h2>
            <div className="meta-row">
              {report.metadata?.author !== 'Unknown' && report.metadata?.author &&
                !report.metadata.author.includes('WhatsApp') && (
                <div className="meta-item">Author<span>{report.metadata.author}</span></div>
              )}
              {report.metadata?.date !== 'Unknown' && (
                <div className="meta-item">Date<span>{report.metadata.date}</span></div>
              )}
            </div>
            <p className="report-summary">{report.summary}</p>
          </div>

          {/* ── Copy Report Button ── */}
          <CopyReportButton report={report} inputType={inputType} />

          {/* WhatsApp Message Details */}
          {isText && <TextMetaCard meta={report.text_metadata} />}

          {/* Key Findings */}
          <KeyFindings findings={report.key_findings} />

          {/* Trust Score Gauge */}
          <div className="score-section">
            <p style={{ fontSize: '.72rem', letterSpacing: '.1em', textTransform: 'uppercase',
              color: 'var(--text-muted)', marginBottom: 8 }}>Trust Score</p>
            <TrustGauge score={report.score} />
            <div className="verdict-label" style={{ color: tierColor }}>
              {report.verdict}
            </div>
          </div>

          {/* Dimensions Grid */}
          <div className="dimensions">

            {/* Factuality */}
            <div className="dim-card">
              <div className="dim-header">
                <span className="dim-label">Factuality</span>
                <span className="dim-weight">{dim.factuality?.weight}</span>
              </div>
              <div className="dim-value">{Math.round((dim.factuality?.value ?? 0) * 100)}%</div>
              <FactualityBar value={dim.factuality?.value ?? 0.5} />
              <p className="dim-explanation" style={{ marginTop: 10 }}>
                {dim.factuality?.explanation}
              </p>
            </div>

            {/* Political Bias */}
            <div className="dim-card">
              <div className="dim-header">
                <span className="dim-label">Political Bias</span>
                <span className="dim-weight">{dim.bias?.weight}</span>
              </div>
              <div className="dim-value">{dim.bias?.value}</div>
              <BiasSlider biasObj={dim.bias} />
              <p className="dim-explanation" style={{ marginTop: 8 }}>
                {dim.bias?.explanation}
              </p>
            </div>

            {/* Intent */}
            <div className="dim-card">
              <div className="dim-header">
                <span className="dim-label">Intent</span>
                <span className="dim-weight">{dim.intent?.weight}</span>
              </div>
              <div className="dim-value" style={{ marginBottom: 12 }}>{dim.intent?.value}</div>
              <IntentChip intent={dim.intent?.value} />
              <p className="dim-explanation" style={{ marginTop: 12 }}>
                {dim.intent?.explanation}
              </p>
            </div>

            {/* Emotion */}
            <div className="dim-card">
              <div className="dim-header">
                <span className="dim-label">Emotion</span>
                <span className="dim-weight">{dim.emotion?.weight}</span>
              </div>
              <div className="dim-value" style={{ marginBottom: 12, textTransform: 'capitalize' }}>{dim.emotion?.value}</div>
              <EmotionChip emotion={dim.emotion?.value} />
              <p className="dim-explanation" style={{ marginTop: 12 }}>
                {dim.emotion?.explanation}
              </p>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
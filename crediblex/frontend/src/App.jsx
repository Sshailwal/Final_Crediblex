import { useState, useEffect } from "react";

import UrlInput from "./components/UrlInput";
import TrustGauge from "./components/TrustGauge";
import BiasSlider from "./components/BiasSlider";
import { FactualityBar, IntentChip, EmotionChip } from "./components/Badges";
import History from "./components/History";
import Navbar from "./components/Navbar";

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:7860';
const MAX_HISTORY = 5;

const TIER_COLOR = (score) => {
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#84cc16";
  if (score >= 40) return "#eab308";
  if (score >= 20) return "#f97316";
  return "#ef4444";
};

// Report and HistoryItem shapes are dynamic objects returned from the API

function CopyReportButton({ report, inputType }) {
  const [copied, setCopied] = useState(false);

  const buildText = () => {
    const dim = report?.dimensions || {};
    const findings = (report.key_findings || [])
      .map((f) => `  • ${f.text}`)
      .join("\n");
    const source =
      inputType === "text"
        ? "[Pasted Text]"
        : report.metadata?.title || "Unknown";

    return [
      "📊 CredibleX Analysis Report",
      "─".repeat(34),
      `📰 Title   : ${source}`,
      `⭐ Score   : ${report.score}/100`,
      `✅ Verdict : ${report.verdict}`,
      "",
      "📊 Dimensions:",
      `  🧾 Factuality : ${Math.round((dim.factuality?.value ?? 0) * 100)}%`,
      `  ⚖️  Bias       : ${dim.bias?.value ?? "N/A"}`,
      `  🎯 Intent     : ${dim.intent?.value ?? "N/A"}`,
      `  😤 Emotion    : ${dim.emotion?.value ?? "N/A"}`,
      "",
      findings ? `📋 Key Findings:\n${findings}` : "",
      "",
      `🤖 Summary:\n${report.summary || "N/A"}`,
      "",
      "─".repeat(34),
      `Analyzed by CredibleX • ${new Date().toLocaleString()}`,
    ]
      .filter((line) => line !== null)
      .join("\n");
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(buildText());
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      alert("Could not copy — please copy manually.");
    }
  };

  return (
    <button
      onClick={handleCopy}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        margin: "0 auto 20px auto",
        padding: "9px 20px",
        background: copied ? "rgba(34,197,94,0.12)" : "rgba(108,99,255,0.12)",
        border: `1px solid ${copied ? "rgba(34,197,94,0.35)" : "rgba(108,99,255,0.35)"}`,
        borderRadius: 10,
        color: copied ? "#4ade80" : "#a78bfa",
        fontSize: "0.82rem",
        fontWeight: 600,
        cursor: "pointer",
        transition: "all 0.2s",
      }}
    >
      {copied ? "✅ Copied!" : "📋 Copy Report"}
    </button>
  );
}

function KeyFindings({ findings }) {
  if (!findings || findings.length === 0) return null;

  const colorMap = {
    good: {
      bg: "rgba(34,197,94,.08)",
      border: "rgba(34,197,94,.25)",
      text: "#4ade80",
    },
    warn: {
      bg: "rgba(234,179,8,.08)",
      border: "rgba(234,179,8,.25)",
      text: "#fde047",
    },
    bad: {
      bg: "rgba(239,68,68,.08)",
      border: "rgba(239,68,68,.25)",
      text: "#f87171",
    },
    info: {
      bg: "rgba(108,99,255,.08)",
      border: "rgba(108,99,255,.25)",
      text: "#a78bfa",
    },
  };

  return (
    <div className="key-findings-card">
      <div className="findings-header">📋 Key Findings</div>
      <ul className="findings-list">
        {findings.map((f, i) => {
          const c = colorMap[f.type] || colorMap.info;
          return (
            <li
              key={i}
              className="finding-item"
              style={{ background: c.bg, borderLeft: `3px solid ${c.border}` }}
            >
              <span className="finding-icon">{f.icon}</span>
              <span className="finding-text" style={{ color: c.text }}>
                {f.text}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function TextMetaCard({ meta }) {
  if (!meta) return null;
  return (
    <div className="text-meta-card">
      <div className="findings-header">📱 Message Details</div>
      <div className="text-meta-grid">
        <div className="text-meta-item">
          <span className="text-meta-label">Word Count</span>
          <span className="text-meta-value">
            {meta.word_count?.toLocaleString()}
          </span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Reading Time</span>
          <span className="text-meta-value">{meta.estimated_read_time}</span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Contains URL</span>
          <span
            className="text-meta-value"
            style={{ color: meta.contains_url ? "#fde047" : "#4ade80" }}
          >
            {meta.contains_url ? "⚠️ Yes" : "✅ No"}
          </span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Looks Like Forward</span>
          <span
            className="text-meta-value"
            style={{ color: meta.looks_like_forward ? "#f87171" : "#4ade80" }}
          >
            {meta.looks_like_forward ? "🚨 Yes — be cautious" : "✅ No"}
          </span>
        </div>
        <div className="text-meta-item">
          <span className="text-meta-label">Numeric Claims</span>
          <span className="text-meta-value">{meta.numeric_claims_count}</span>
        </div>
        {meta.likely_non_english && (
          <div className="text-meta-item" style={{ gridColumn: "1 / -1" }}>
            <span className="text-meta-label">Language</span>
            <span className="text-meta-value" style={{ color: "#fde047" }}>
              ⚠️ Possibly non-English — model accuracy may be reduced
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

const safeNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatPercent = (value) => `${Math.round(safeNumber(value) * 100)}%`;

function DashboardCard({ className = "", eyebrow, title, value, children }) {
  return (
    <section className={`result-card ${className}`}>
      <div className="result-card-header">
        <span className="result-card-eyebrow">{eyebrow}</span>
        <h3>{title}</h3>
      </div>
      {value && <div className="result-card-value">{value}</div>}
      {children}
    </section>
  );
}

function ReportDashboard({ report, inputType }) {
  const dimensions = report?.dimensions || {};
  const factuality = dimensions.factuality || {};
  const bias = dimensions.bias || {};
  const intent = dimensions.intent || {};
  const emotion = dimensions.emotion || {};
  const metadata = report?.metadata || {};
  const score = safeNumber(report?.score);
  const tierColor = TIER_COLOR(score);
  const title =
    metadata.title || (inputType === "text" ? "Message Analysis" : "Article Analysis");
  const author = metadata.author && metadata.author !== "Unknown" ? metadata.author : "";
  const date = metadata.date && metadata.date !== "Unknown" ? metadata.date : "";
  const emotionValue = emotion.value || emotion.top?.[0]?.label || "Neutral";

  return (
    <div className="result-dashboard">
      <section className="result-overview">
        {inputType === "text" && (
          <div className="whatsapp-badge">💬 WhatsApp / Text Analysis</div>
        )}
        <div>
          <span className="result-kicker">Credibility Report</span>
          <h2>{title}</h2>
        </div>
        {(author || date) && (
          <div className="result-meta-row">
            {author && (
              <span>
                Author <strong>{author}</strong>
              </span>
            )}
            {date && (
              <span>
                Date <strong>{date}</strong>
              </span>
            )}
          </div>
        )}
        {report.extraction_warning && (
          <div className="extraction-warning">{report.extraction_warning}</div>
        )}
      </section>

      <div className="result-grid">
        <DashboardCard
          className="trust-card"
          eyebrow="Major score"
          title="Trust / Credibility"
        >
          <TrustGauge score={score} />
          <div className="verdict-pill" style={{ color: tierColor }}>
            {report?.verdict || "Unknown verdict"}
          </div>
        </DashboardCard>

        <DashboardCard
          eyebrow="Factuality"
          title="Factuality Percentage"
          value={formatPercent(factuality.value)}
        >
          <FactualityBar value={safeNumber(factuality.value)} />
          <p>{factuality.explanation || "No factuality explanation was returned."}</p>
        </DashboardCard>

        <DashboardCard
          className="bias-card"
          eyebrow="Political leaning"
          title="Political Bias"
        >
          <BiasSlider biasObj={bias} />
          <p>
            {bias.explanation ||
              "No political bias explanation was returned by the backend."}
          </p>
        </DashboardCard>

        <DashboardCard
          eyebrow="Classification"
          title="Intent"
          value={intent.value || "Unknown"}
        >
          <IntentChip intent={intent.value || "Unknown"} />
          <p>{intent.explanation || "No intent explanation was returned."}</p>
        </DashboardCard>

        <DashboardCard
          eyebrow="Tone"
          title="Emotion"
          value={emotionValue}
        >
          <EmotionChip emotion={emotionValue} />
          <p>{emotion.explanation || "No dominant emotion was returned."}</p>
        </DashboardCard>

        <DashboardCard
          className="summary-card"
          eyebrow="Article summary"
          title="Summary"
        >
          <p className="summary-text">{report?.summary || "No summary available."}</p>
        </DashboardCard>
      </div>
    </div>
  );
}

export default function Page() {
  const [state, setState] = useState("idle");
  const [report, setReport] = useState(null);
  const [errMsg, setErrMsg] = useState("");
  const [inputType, setInputType] = useState("url");
  const [history, setHistory] = useState([]);

  // Load history from localStorage on startup
  useEffect(() => {
    try {
      const saved = localStorage.getItem("crediblex_history");
      if (saved) setHistory(JSON.parse(saved));
    } catch {
      setHistory([]);
    }
  }, []);

  // Save a new result to history
  const saveToHistory = (report, type, value) => {
    try {
      const entry = {
        type,
        value,
        title:
          report.metadata?.title ||
          (type === "text" ? value.slice(0, 60) : value) ||
          "Analysis",
        score: report.score ?? 0,
        verdict: report.verdict || "Unknown",
        report,
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setHistory((prev) => {
        const updated = [entry, ...prev].slice(0, MAX_HISTORY);
        try {
          localStorage.setItem("crediblex_history", JSON.stringify(updated));
        } catch (e) {
          console.error("History save failed:", e);
        }
        return updated;
      });
    } catch (err) {
      console.error("Error preparing history entry:", err);
    }
  };

  // Clear history
  const clearHistory = () => {
    localStorage.removeItem("crediblex_history");
    setHistory([]);
  };

  // Click a history item to restore that report
  const handleHistorySelect = (item) => {
    setReport(item.report);
    setInputType(item.type);
    setState("success");
  };

  const analyze = async ({ type, value }) => {
    setState("loading");
    setReport(null);
    setErrMsg("");
    setInputType(type);

    try {
      const endpoint =
        type === "url" ? `${API}/analyze-url` : `${API}/analyze-text`;
      const body = type === "url" ? { url: value } : { text: value };

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        const msg =
          data?.detail?.message || data?.detail || "Unknown error from server.";
        throw new Error(msg);
      }
      setReport(data);
      setState("success");
      saveToHistory(data, type, value);
    } catch (err) {
      setErrMsg(
        err instanceof Error
          ? err.message
          : "Network error — is the API running on port 7860?",
      );
      setState("error");
    }
  };

  const isText = inputType === "text";

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        background: "#F5F5F5",
        color: "#F3F4F6",
      }}
    >
      
      {/* Main Content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {/* Content Area */}
        <Navbar/>
        <main
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "32px 20px",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: "100%",
              maxWidth: "1200px",
            }}
          >
            {/* Hero Section */}
            <div
              style={{
                textAlign: "center",
                marginBottom: "40px",
              }}
            >
              <h1
                style={{
                  fontSize: "2.7rem",
                  fontWeight: 800,
                  marginBottom: "14px",
                  lineHeight: 1.1,
                  color: "black"
                }}
              >
                News Article{" "}
                <span style={{ color: "#EF4444" }}>Credibility</span>
              </h1>

              <p
                style={{
                  fontSize: "1.30rem",
                  color: "black",
                  maxWidth: "800px",
                  margin: "0 auto",
                  lineHeight: 3,
                }}
              >
                Get instant insights into factuality, bias, intent, and
                emotional tone of any article or text.
              </p>
            </div>

            {/* Input Component */}
            <div
              style={{
                width: "100%",
                display: "flex",
                justifyContent: "center",
                marginBottom: "28px",
              }}
            >
              <div
                style={{
                  width: "100%",
                  maxWidth: "850px",
                }}
              >
                <UrlInput onSubmit={analyze} loading={state === "loading"} />
              </div>
            </div>

            {/* History */}
            <div style={{ marginBottom: "24px" }}>
              <History
                history={history}
                onSelect={handleHistorySelect}
                onClear={clearHistory}
              />
            </div>

            {/* Status Area */}
            <div
              className="status-area"
              style={{
                marginBottom: "24px",
              }}
            >
              {state === "loading" && (
                <>
                  <div
                    className="loading-bar"
                    style={{ animationDuration: "2s" }}
                  />
                  <p className="loading-text">
                    {isText
                      ? "Fact-checking message — running model inference…"
                      : "Scraping article & running model inference…"}
                  </p>
                </>
              )}

              {state === "error" && (
                <div className="error-box">
                  <strong>Analysis Failed</strong>
                  {errMsg}
                </div>
              )}
            </div>

            {/* Report */}
            {state === "success" && report && (
              <div
                className="report"
                style={{
                  width: "100%",
                  display: "flex",
                  flexDirection: "column",
                  gap: "24px",
                }}
              >
                <ReportDashboard report={report} inputType={inputType} />
                <CopyReportButton report={report} inputType={inputType} />
                {isText && <TextMetaCard meta={report.text_metadata} />}
                <KeyFindings findings={report.key_findings} />
              </div>
            )}
          </div>
        </main>
        {/* Footer */}
      <footer
  style={{
    borderTop: "1px solid #fecaca",
    padding: "24px 20px 12px",
    background: "#fff5f5",
    marginTop: "40px",
  }}
>
  <div
    style={{
      maxWidth: "1100px",
      margin: "0 auto",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      flexWrap: "wrap",
      gap: "16px",
    }}
  >
    {/* Brand */}
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <div
        style={{
          width: "28px",
          height: "28px",
          borderRadius: "7px",
          background: "linear-gradient(135deg, #ef4444, #dc2626)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          fontWeight: "700",
          fontSize: "13px",
        }}
      >
        C
      </div>

      <span style={{ fontSize: "16px", fontWeight: "700", color: "#111827" }}>
        CredibleX
      </span>
    </div>

    {/* Center text */}
    <p style={{ fontSize: "13px", color: "#6b7280", margin: 0 }}>
      AI-powered article credibility analysis
    </p>

    {/* Bottom right */}
    <span style={{ fontSize: "12px", color: "#9ca3af" }}>
      © 2026 CredibleX
    </span>
  </div>
</footer>
      </div>
    </div>
  );
}

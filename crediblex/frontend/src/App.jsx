import { useState, useEffect } from "react";

import UrlInput from "./components/UrlInput";
import TrustGauge from "./components/TrustGauge";
import BiasSlider from "./components/BiasSlider";
import { FactualityBar, IntentChip, EmotionChip } from "./components/Badges";
import History from "./components/History";
import Navbar from "./components/Navbar";

const API = "http://127.0.0.1:7860";
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

  const tierColor = report ? TIER_COLOR(report.score) : "#6c63ff";
  const dim = report?.dimensions || {};
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
                {/* Metadata */}
                <div
                  className="report-meta"
                  style={{
                    padding: "28px",
                    borderRadius: "20px",
                  }}
                >
                  {isText && (
                    <div className="whatsapp-badge">
                      💬 WhatsApp / Text Analysis
                    </div>
                  )}

                  <h2
                    style={{
                      marginTop: "12px",
                      marginBottom: "18px",
                    }}
                  >
                    {report.metadata?.title ||
                      (isText ? "Message Analysis" : "Article")}
                  </h2>

                  <div
                    className="meta-row"
                    style={{
                      display: "flex",
                      gap: "16px",
                      flexWrap: "wrap",
                      marginBottom: "16px",
                    }}
                  >
                    {report.metadata?.author &&
                      report.metadata.author !== "Unknown" &&
                      !report.metadata.author.includes("WhatsApp") && (
                        <div className="meta-item">
                          Author
                          <span>{report.metadata.author}</span>
                        </div>
                      )}

                    {report.metadata?.date &&
                      report.metadata.date !== "Unknown" && (
                        <div className="meta-item">
                          Date
                          <span>{report.metadata.date}</span>
                        </div>
                      )}
                  </div>

                  <p
                    className="report-summary"
                    style={{
                      lineHeight: 1.8,
                    }}
                  >
                    {report.summary}
                  </p>
                </div>

                {/* Copy Button */}
                <CopyReportButton report={report} inputType={inputType} />

                {/* WhatsApp Card */}
                {isText && <TextMetaCard meta={report.text_metadata} />}

                {/* Findings */}
                <KeyFindings findings={report.key_findings} />

                {/* Score */}
                <div
                  className="score-section"
                  style={{
                    textAlign: "center",
                    padding: "32px 20px",
                  }}
                >
                  <p
                    style={{
                      fontSize: ".72rem",
                      letterSpacing: ".1em",
                      textTransform: "uppercase",
                      color: "#9ca3af",
                      marginBottom: 8,
                    }}
                  >
                    Trust Score
                  </p>

                  <TrustGauge score={report.score} />

                  <div
                    className="verdict-label"
                    style={{
                      color: tierColor,
                      marginTop: "16px",
                    }}
                  >
                    {report.verdict}
                  </div>
                </div>

                {/* Dimensions */}
                <div
                  className="dimensions"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                    gap: "20px",
                    alignItems: "stretch",
                  }}
                >
                  {/* CARD */}
                  {[
                    {
                      label: "Factuality",
                      value: `${Math.round(
                        (dim.factuality?.value ?? 0) * 100,
                      )}%`,
                      explanation: dim.factuality?.explanation,
                      component: (
                        <FactualityBar value={dim.factuality?.value ?? 0.5} />
                      ),
                      weight: dim.factuality?.weight,
                    },
                    {
                      label: "Political Bias",
                      value: dim.bias?.value,
                      explanation: dim.bias?.explanation,
                      component: <BiasSlider biasObj={dim.bias} />,
                      weight: dim.bias?.weight,
                    },
                    {
                      label: "Intent",
                      value: dim.intent?.value,
                      explanation: dim.intent?.explanation,
                      component: <IntentChip intent={dim.intent?.value} />,
                      weight: dim.intent?.weight,
                    },
                    {
                      label: "Emotion",
                      value: dim.emotion?.value,
                      explanation: dim.emotion?.explanation,
                      component: <EmotionChip emotion={dim.emotion?.value} />,
                      weight: dim.emotion?.weight,
                    },
                  ].map((item, idx) => (
                    <div
                      key={idx}
                      className="dim-card"
                      style={{
                        padding: "24px",
                        borderRadius: "18px",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        minHeight: "260px",
                      }}
                    >
                      <div className="dim-header">
                        <span className="dim-label">{item.label}</span>
                        <span className="dim-weight">{item.weight}</span>
                      </div>

                      <div
                        className="dim-value"
                        style={{
                          margin: "14px 0",
                        }}
                      >
                        {item.value}
                      </div>

                      {item.component}

                      <p
                        className="dim-explanation"
                        style={{
                          marginTop: "16px",
                          lineHeight: 1.7,
                        }}
                      >
                        {item.explanation}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feature Cards */}
           {state === "idle" && (
  <div
    style={{
      marginTop: "70px",
      marginBottom: "70px",
      width: "100%",
    }}
  >
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
        gap: "24px",
        alignItems: "stretch",
      }}
    >
      {[
        {
          icon: "📊",
          title: "Comprehensive Analysis",
          desc: "Get detailed insights on factuality, bias, intent, and emotion.",
        },
        {
          icon: "⚡",
          title: "Instant Results",
          desc: "Get analysis in seconds, not hours. Powered by advanced AI.",
        },
        {
          icon: "📈",
          title: "Track History",
          desc: "Keep a record of all your analyses for future reference.",
        },
      ].map((card, idx) => (
        <div
          key={idx}
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "18px",
            padding: "34px 28px",
            textAlign: "center",
            minHeight: "220px",

            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",

            boxShadow: "0 2px 10px rgba(0,0,0,0.04)",
            transition: "all 0.25s ease",
            cursor: "default",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-4px)";
            e.currentTarget.style.boxShadow =
              "0 8px 24px rgba(0,0,0,0.08)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0px)";
            e.currentTarget.style.boxShadow =
              "0 2px 10px rgba(0,0,0,0.04)";
          }}
        >
          <div
            style={{
              fontSize: "2.8rem",
              marginBottom: "18px",
              lineHeight: 1,
            }}
          >
            {card.icon}
          </div>

          <h3
            style={{
              fontSize: "1.2rem",
              fontWeight: "700",
              color: "#111827",
              marginBottom: "12px",
            }}
          >
            {card.title}
          </h3>

          <p
            style={{
              fontSize: ".95rem",
              color: "#6b7280",
              lineHeight: 1.7,
              maxWidth: "260px",
            }}
          >
            {card.desc}
          </p>
        </div>
      ))}
    </div>
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

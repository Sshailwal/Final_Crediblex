/* BiasSlider.jsx — Five-point political bias visualization */
export default function BiasSlider({ biasObj }) {
  const rawLabel = String(biasObj?.label || biasObj?.value || "center")
    .toLowerCase()
    .replace(/\s+/g, "_");

  const scale = {
    far_left: { label: "Far Left", position: 0, color: "#dc2626" },
    left: { label: "Left", position: 25, color: "#f97316" },
    slightly_left: { label: "Slightly Left", position: 37.5, color: "#f59e0b" },
    center: { label: "Center", position: 50, color: "#22c55e" },
    slightly_right: { label: "Slightly Right", position: 62.5, color: "#06b6d4" },
    right: { label: "Right", position: 75, color: "#3b82f6" },
    far_right: { label: "Far Right", position: 100, color: "#4338ca" },
  };

  const active = scale[rawLabel] || scale.center;
  const labels = ["Far Left", "Left", "Center", "Right", "Far Right"];

  return (
    <div className="bias-meter">
      <div className="bias-current" style={{ color: active.color }}>
        {active.label}
      </div>
      <div className="bias-track">
        <div
          className="bias-thumb"
          style={{
            left: `${active.position}%`,
            background: active.color,
            boxShadow: `0 0 0 4px ${active.color}22, 0 8px 18px rgba(0,0,0,.12)`,
          }}
        />
      </div>
      <div className="bias-labels">
        {labels.map((label) => (
          <span
            key={label}
            style={{
              color: label === active.label ? active.color : undefined,
              fontWeight: label === active.label ? 700 : undefined,
            }}
          >
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

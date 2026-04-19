/* BiasSlider.jsx — Left-Center-Right slider visualization */
export default function BiasSlider({ biasObj }) {
  const label = biasObj?.value || 'Center';
  const left = biasObj?.position ?? 50;

  let thumbColor = '#9ca3af';
  if (label.includes('Left'))  thumbColor = '#ef4444';
  if (label === 'Center')      thumbColor = '#22c55e';
  if (label.includes('Right')) thumbColor = '#3b82f6';

  return (
    <div>
      <div className="bias-track">
        <div
          className="bias-thumb"
          style={{ left: `${left}%`, background: thumbColor, boxShadow: `0 0 0 3px ${thumbColor}44, 0 2px 8px rgba(0,0,0,.5)` }}
        />
      </div>
      <div className="bias-labels">
        <span style={{ color: label.includes('Left')   ? '#ef4444' : undefined }}>Left</span>
        <span style={{ color: label === 'Center'       ? '#22c55e' : undefined }}>Center</span>
        <span style={{ color: label.includes('Right')  ? '#3b82f6' : undefined }}>Right</span>
      </div>
    </div>
  );
}

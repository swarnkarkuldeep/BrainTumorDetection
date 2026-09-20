const LOW_CONFIDENCE_THRESHOLD = 0.6;

const VERDICT_TEXT = {
  no: "No tumor detected",
  yes: "Tumor detected",
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function ResultPanel({ status, result, error, tone }) {
  if (status === "error") {
    return (
      <div className="result-panel result-panel--error">
        <p>{error}</p>
      </div>
    );
  }

  if (status !== "done" || !result) {
    return null;
  }

  const verdict = VERDICT_TEXT[result.predicted_class] ?? result.predicted_class;
  const isLowConfidence = result.confidence < LOW_CONFIDENCE_THRESHOLD;
  const probabilities = Object.entries(result.all_probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="result-panel">
      <div className="result-panel__verdict">
        <h2 className={`result-panel__label result-panel__label--${tone}`}>{verdict}</h2>
        <span className="result-panel__confidence">{formatPercent(result.confidence)}</span>
      </div>

      <div className="result-panel__bars">
        {probabilities.map(([label, value]) => (
          <div className="probability-row" key={label}>
            <span className="probability-row__label">{label}</span>
            <div className="probability-row__track">
              <div
                className={`probability-row__fill probability-row__fill--${label}`}
                style={{ width: `${value * 100}%` }}
              />
            </div>
            <span className="probability-row__value">{formatPercent(value)}</span>
          </div>
        ))}
      </div>

      {isLowConfidence && (
        <p className="result-panel__caution">
          Confidence is low. Treat this reading as inconclusive rather than a diagnosis.
        </p>
      )}
    </div>
  );
}

import PropTypes from 'prop-types';

function EnergyLogPanel({ entries, onClear, onSetThreshold }) {
  const canSetThreshold = entries.length > 5;

  return (
    <div className="energy-log">
      <h2 className="section-title">Energy Log</h2>
      {entries.length === 0 ? (
        <p className="empty-state">No entries yet. Start recording to populate the log.</p>
      ) : (
        <ul className="energy-level-list">
          {entries.map((energy, index) => (
            <li key={`${energy}-${index}`} className="energy-level-item">
              Energy Level: {energy.toFixed(5)}
            </li>
          ))}
        </ul>
      )}
      <div className="log-buttons">
        <button className="clear-log" onClick={onClear}>
          Clear Log
        </button>
        <button className="set-threshold" onClick={onSetThreshold} disabled={!canSetThreshold}>
          Set Threshold
        </button>
      </div>
    </div>
  );
}

EnergyLogPanel.propTypes = {
  entries: PropTypes.arrayOf(PropTypes.number).isRequired,
  onClear: PropTypes.func.isRequired,
  onSetThreshold: PropTypes.func.isRequired,
};

export default EnergyLogPanel;

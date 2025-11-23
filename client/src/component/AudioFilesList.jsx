import PropTypes from 'prop-types';
import Waveform from './Waveform';

const formatConfidence = (confidence) => {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) {
    return '0%';
  }
  return `${Math.round(confidence * 100)}%`;
};

function AudioFilesList({
  files,
  apiBaseUrl,
  authToken = null,
  onClassify,
  onDelete,
  classifyingId = null,
  onDeleteAll,
}) {
  return (
    <div className="audio-files-section">
      <div className="audio-files-header">
        <h3 className="section-subtitle">Audio Files</h3>
        <button className="delete-all-button" onClick={onDeleteAll}>
          Delete All
        </button>
      </div>
      {files.length === 0 ? (
        <p className="empty-state">No recordings yet. Start recording to capture your sleep audio.</p>
      ) : (
        <ul className="audio-files-list">
          {files.map((file) => {
            const classification = file.classification ?? 'unclassified';
            return (
              <li key={file.file_id} className="audio-file-item">
                <div className="file-header">
                  <span className="file-name">{file.filename}</span>
                  {classification !== 'unclassified' && (
                    <span className={`classification-badge ${classification}`}>
                      Class: {classification} ({formatConfidence(file.confidence)})
                    </span>
                  )}
                </div>
                <div className="file-actions">
                  <Waveform audioUrl={`${apiBaseUrl}/${file.audio_url}`} token={authToken} />
                  <div className="file-buttons">
                    <button
                      onClick={() => onClassify(file.file_id)}
                      className="classify-button"
                      disabled={classifyingId === file.file_id}
                    >
                      {classifyingId === file.file_id ? 'Classifying…' : 'Classify'}
                    </button>
                    <button onClick={() => onDelete(file.file_id)} className="delete-button">
                      Delete
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

AudioFilesList.propTypes = {
  files: PropTypes.arrayOf(
    PropTypes.shape({
      file_id: PropTypes.string.isRequired,
      filename: PropTypes.string.isRequired,
      audio_url: PropTypes.string.isRequired,
      classification: PropTypes.string,
      confidence: PropTypes.number,
    })
  ).isRequired,
  apiBaseUrl: PropTypes.string.isRequired,
  authToken: PropTypes.string,
  onClassify: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
  onDeleteAll: PropTypes.func.isRequired,
  classifyingId: PropTypes.string,
};

export default AudioFilesList;

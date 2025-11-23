function CalibrationInstructions() {
  return (
    <div className="calibrate-manual">
      <h2 className="section-title">Calibration Instructions</h2>
      <ol className="calibrate-instructions">
        <li>Ensure an environment with minimal noise.</li>
        <li>Press the &quot;Start Recording&quot; button.</li>
        <li>Keep the room quiet so only idle noise is captured.</li>
        <li>Allow the Energy Log to populate with entries.</li>
        <li>Clear the log if spikes occur due to unexpected noise.</li>
        <li>Press &quot;Set Threshold&quot; once at least five entries are logged.</li>
        <li>Alternatively, tweak the threshold manually based on observations.</li>
      </ol>
    </div>
  );
}

export default CalibrationInstructions;

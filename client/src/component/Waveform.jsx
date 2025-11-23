import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { useWavesurfer } from '@wavesurfer/react';
import Timeline from 'wavesurfer.js/dist/plugins/timeline.esm.js';
import axios from 'axios';

const formatTime = (seconds = 0) =>
  [seconds / 60, seconds % 60]
    .map((value) => `0${Math.floor(value)}`.slice(-2))
    .join(':');

function Waveform({ audioUrl, token = null }) {
  const waveformRef = useRef(null);
  const [blobUrl, setBlobUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!audioUrl || !token) {
      return undefined;
    }

    const controller = new AbortController();

    const fetchAudio = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get(audioUrl, {
          responseType: 'blob',
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        });
        const url = URL.createObjectURL(response.data);
        setBlobUrl((prev) => {
          if (prev) {
            URL.revokeObjectURL(prev);
          }
          return url;
        });
      } catch {
        setBlobUrl((prev) => {
          if (prev) {
            URL.revokeObjectURL(prev);
          }
          return null;
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchAudio();

    return () => {
      controller.abort();
      setBlobUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return null;
      });
    };
  }, [audioUrl, token]);

  const { wavesurfer, isPlaying, currentTime } = useWavesurfer({
    container: waveformRef,
    height: 80,
    waveColor: 'violet',
    progressColor: 'purple',
    cursorColor: 'navy',
    url: blobUrl,
    plugins: useMemo(() => [Timeline.create()], []),
  });

  const handlePlayPause = useCallback(() => {
    if (wavesurfer) {
      wavesurfer.playPause();
    }
  }, [wavesurfer]);

  return (
    <div className="waveform-wrapper">
      <div ref={waveformRef} />
      <p>{isLoading ? 'Loading…' : `Current time: ${formatTime(currentTime)}`}</p>
      <button onClick={handlePlayPause} style={{ minWidth: '5em' }} className="play-button" disabled={!blobUrl}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
}

Waveform.propTypes = {
  audioUrl: PropTypes.string.isRequired,
  token: PropTypes.string,
};

export default Waveform;

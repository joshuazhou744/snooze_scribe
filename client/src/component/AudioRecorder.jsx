import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { Link } from 'react-router-dom';
import RecordRTC from 'recordrtc';
import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';
import AudioFilesList from './AudioFilesList';
import EnergyLogPanel from './EnergyLogPanel';
import CalibrationInstructions from './CalibrationInstructions';
import {
  classifyAudioFile,
  deleteAllAudioFiles,
  deleteAudioFile,
  fetchAudioFiles,
  uploadAudioFile,
} from '../services/audioService';
import './AudioRecorder.css';

const API_BASE_URL = import.meta.env.VITE_API_URL;
const AUTH0_AUDIENCE = import.meta.env.VITE_AUTH0_AUDIENCE;
const FETCH_INTERVAL_MS = 20_000;
const AUTO_DELETE_INTERVAL_MS = 120 * 60 * 1000;
const MAX_AUDIO_FILES = 500;
const MAX_ENERGY_LOG_ENTRIES = 50;
const EXPIRY_THRESHOLD_DAYS = 10;

const parseRecordingDate = (filename) => {
  const match = filename.match(/sleep_recording_(\d{1,2})-(\d{1,2})-(\d{4})_/);
  if (!match) {
    return null;
  }
  const [, month, day, year] = match;
  return new Date(Number(year), Number(month) - 1, Number(day));
};

const formatEnergyValue = (value) => Number(value.toFixed(5));

function AudioRecorder() {
  const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();
  const [audioFiles, setAudioFiles] = useState([]);
  const [energyThreshold, setEnergyThreshold] = useState(0.05);
  const [energyLog, setEnergyLog] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [classifyingId, setClassifyingId] = useState(null);
  const [authToken, setAuthToken] = useState(null);

  const recorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const isRecordingRef = useRef(false);
  const wakeLockRef = useRef(null);

  const getFileName = useCallback(() => {
    const now = new Date();
    const formattedDate = `${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}_${now.getHours()}h-${now.getMinutes()}m-${now.getSeconds()}s`;
    return `sleep_recording_${formattedDate}.mp4`;
  }, []);

  const fetchToken = useCallback(async () => {
    const token = await getAccessTokenSilently({ audience: AUTH0_AUDIENCE });
    setAuthToken(token);
    return token;
  }, [getAccessTokenSilently]);

  const withAuth = useCallback(
    async (fn) => {
      const token = await fetchToken();
      return fn(token);
    },
    [fetchToken]
  );

  const loadAudioFiles = useCallback(async () => {
    if (!isAuthenticated || !user) {
      return;
    }
    const files = await withAuth((token) => fetchAudioFiles(token));
    setAudioFiles(files);
  }, [isAuthenticated, user, withAuth]);

  const requestWakeLock = useCallback(async () => {
    if (!('wakeLock' in navigator) || typeof navigator.wakeLock?.request !== 'function') {
      return;
    }
    wakeLockRef.current = await navigator.wakeLock.request('screen');
  }, []);

  const releaseWakeLock = useCallback(async () => {
    if (!wakeLockRef.current) {
      return;
    }
    await wakeLockRef.current.release();
    wakeLockRef.current = null;
  }, []);

  const calculateRMS = useCallback((blob) => {
    return new Promise((resolve, reject) => {
      const audioContext = new window.AudioContext();
      const reader = new FileReader();

      reader.onload = (event) => {
        audioContext
          .decodeAudioData(event.target.result)
          .then((audioBuffer) => {
            const channelData = audioBuffer.getChannelData(0);
            let sum = 0;
            for (let i = 0; i < channelData.length; i += 1) {
              sum += channelData[i] * channelData[i];
            }
            const rms = Math.sqrt(sum / channelData.length);
            audioContext.close();
            resolve(rms);
          })
          .catch((error) => {
            audioContext.close();
            reject(error);
          });
      };

      reader.onerror = reject;
      reader.readAsArrayBuffer(blob);
    });
  }, []);

  const uploadAudioChunk = useCallback(
    async (audioBlob) => {
      await withAuth((token) => uploadAudioFile(token, audioBlob, getFileName()));
      await loadAudioFiles();
    },
    [getFileName, loadAudioFiles, withAuth]
  );

  const processAudioChunk = useCallback(
    async (audioBlob) => {
      const rms = await calculateRMS(audioBlob);
      const energyValue = formatEnergyValue(rms);
      console.log(`[Energy] RMS: ${energyValue}`);
      setEnergyLog((prev) => {
        const trimmed = prev.slice(-MAX_ENERGY_LOG_ENTRIES + 1);
        return [...trimmed, energyValue];
      });

      if (rms <= energyThreshold) {
        return;
      }

      if (audioFiles.length >= MAX_AUDIO_FILES) {
        return;
      }

      await uploadAudioChunk(audioBlob);
    },
    [audioFiles, energyThreshold, calculateRMS, uploadAudioChunk]
  );

  const handleRecorderData = useCallback((blob) => processAudioChunk(blob), [processAudioChunk]);

  const startRecording = useCallback(async () => {
    if (isRecording) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;

    recorderRef.current = new RecordRTC(stream, {
      type: 'audio',
      mimeType: 'audio/mp4;codecs=mp4a.40.2',
      timeSlice: 15000,
      desiredSampRate: 44100,
      numberOfAudioChannels: 1,
      bufferSize: 4096,
      recorderType: RecordRTC.StereoAudioRecorder,
      ondataavailable: (blob) => {
        handleRecorderData(blob).catch(() => {
          // Tolerate chunk errors silently to avoid console noise
        });
      },
    });

    recorderRef.current.startRecording();
    isRecordingRef.current = true;
    setIsRecording(true);
    await requestWakeLock();
  }, [handleRecorderData, isRecording, requestWakeLock]);

  const stopRecording = useCallback(() => {
    if (!isRecordingRef.current) {
      return;
    }

    recorderRef.current?.stopRecording();
    recorderRef.current = null;
    setIsRecording(false);
    isRecordingRef.current = false;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    releaseWakeLock();
  }, [releaseWakeLock]);

  const handleDelete = useCallback(
    async (fileId) => {
      await withAuth((token) => deleteAudioFile(token, fileId));
      setAudioFiles((prev) => prev.filter((file) => file.file_id !== fileId));
    },
    [withAuth]
  );

  const deleteExpiredFiles = useCallback(async () => {
    if (!audioFiles.length) {
      return;
    }
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    const expiredFiles = audioFiles.filter((file) => {
      const recordedDate = parseRecordingDate(file.filename);
      if (!recordedDate) {
        return false;
      }
      const ageInDays = (now - recordedDate.getTime()) / dayMs;
      return ageInDays >= EXPIRY_THRESHOLD_DAYS;
    });

    for (const file of expiredFiles) {
      await handleDelete(file.file_id);
    }
  }, [audioFiles, handleDelete]);

  const handleClassify = useCallback(
    async (fileId) => {
      try {
        setClassifyingId(fileId);
        const result = await withAuth((token) => classifyAudioFile(token, fileId));
        setAudioFiles((prev) =>
          prev.map((file) =>
            file.file_id === fileId
              ? {
                  ...file,
                  classification: result.classification,
                  confidence: result.confidence,
                }
              : file
          )
        );
      } finally {
        setClassifyingId(null);
      }
    },
    [withAuth]
  );

  const handleDeleteAll = useCallback(async () => {
    const confirmed = window.confirm('Delete ALL audio files? This cannot be undone.');
    if (!confirmed) {
      return;
    }
    await withAuth((token) => deleteAllAudioFiles(token));
    setAudioFiles([]);
  }, [withAuth]);

  const clearEnergyLog = useCallback(() => {
    setEnergyLog([]);
  }, []);

  const updateEnergyThreshold = useCallback(() => {
    if (!energyLog.length) {
      return;
    }
    const sum = energyLog.reduce((acc, value) => acc + value, 0);
    const average = sum / energyLog.length;
    setEnergyThreshold(formatEnergyValue(average));
  }, [energyLog]);

  useEffect(() => {
    loadAudioFiles();
    if (!isAuthenticated) {
      return undefined;
    }
    const intervalId = setInterval(loadAudioFiles, FETCH_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [isAuthenticated, loadAudioFiles]);

  useEffect(() => {
    if (!isAuthenticated) {
      return undefined;
    }
    deleteExpiredFiles();
    const intervalId = setInterval(deleteExpiredFiles, AUTO_DELETE_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [deleteExpiredFiles, isAuthenticated]);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  const recorderSection = useMemo(() => {
    if (!isAuthenticated) {
      return (
        <div className="login-prompt">Please login to use Snooze Scribe.</div>
      );
    }

    return (
      <div className="recorder-section">
        <h2 className="section-title">Snooze Scribe</h2>

        <div className="threshold-control">
          <label htmlFor="energyThreshold">Energy Threshold:</label>
          <input
            type="number"
            id="energyThreshold"
            value={energyThreshold}
            placeholder="Energy Threshold"
            onChange={(event) => setEnergyThreshold(parseFloat(event.target.value) || 0)}
            step="0.01"
            min="0"
            max="10"
            className="threshold-input"
          />
        </div>

        <div className="recording-controls">
          <button onClick={startRecording} disabled={isRecording} className="start-button">
            Start Recording
          </button>
          <button onClick={stopRecording} disabled={!isRecording} className="stop-button">
            Stop Recording
          </button>
        </div>

        <AudioFilesList
          files={audioFiles}
          apiBaseUrl={API_BASE_URL}
          authToken={authToken}
          onClassify={handleClassify}
          onDelete={handleDelete}
          onDeleteAll={handleDeleteAll}
          classifyingId={classifyingId}
        />
      </div>
    );
  }, [
    authToken,
    audioFiles,
    classifyingId,
    handleClassify,
    handleDelete,
    handleDeleteAll,
    isAuthenticated,
    isRecording,
    startRecording,
    stopRecording,
    energyThreshold,
  ]);

  return (
    <div className="container">
      <div className="header">
        <Link to="/user-guide">
          <div className="about">
            <button className="manual-button" onClick={stopRecording}>
              User Guide
            </button>
          </div>
        </Link>

        <div className="auth-buttons">
          <LoginButton className="login-button" isAuthenticated={isAuthenticated} />
          <LogoutButton className="logout-button" />
        </div>
      </div>

      {recorderSection}

      {isAuthenticated && (
        <div className="layer">
          <EnergyLogPanel
            entries={energyLog}
            onClear={clearEnergyLog}
            onSetThreshold={updateEnergyThreshold}
          />
          <CalibrationInstructions />
        </div>
      )}
    </div>
  );
}

export default AudioRecorder;

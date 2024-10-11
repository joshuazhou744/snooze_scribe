import { useState, useRef, useEffect } from 'react'
import { useAuth0 } from '@auth0/auth0-react';
import { Link } from 'react-router-dom';
import RecordRTC from 'recordrtc';
import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';
import axios from 'axios'
import Waveform from './Waveform';
import './AudioRecorder.css';

const AudioRecorder = () => {
  const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();
  const [audioFiles, setAudioFiles] = useState([]);
  const [isRecording, setIsRecording] = useState(false)
  const isRecordingRef = useRef(isRecording);
  const [energyThreshold, setEnergyThreshold] = useState(0.05);
  const [expiryThreshold, setExpiryTHreshold] = useState(3);
  const [token, setToken] = useState(null);
  const [energyLog, setEnergyLog] = useState([]);
  const [wakeLock, setWakeLock] = useState(null); // keeps screen on during the night for IOS Mobile Users because Webkit API is terrible and halts all background processes past a grace period on sleep
  const recorderRef = useRef(null);
  const mediaStreamRef = useRef(null)
  const apiUrl = import.meta.env.VITE_API_URL;
  const audience= import.meta.env.VITE_AUTH0_AUDIENCE;

  useEffect(() => {
    fetchAudioFiles()
    const interval = setInterval(fetchAudioFiles, 20000)
    return () => clearInterval(interval)
  }, [isAuthenticated])

  useEffect(() => {
    autoDelete()
  }, [isRecording])

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording])

  const requestWakeLock = async () => {
    try {
      const lock = await navigator.wakeLock.request('screen');
      setWakeLock(lock)
      console.log('wakelock is now active')
    } catch (err) {
      console.error(`${err.name}, ${err.message}`)
    }
  }

  const getFileName = () => {
    const now = new Date()
    const formattedDate = `${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}_${now.getHours()}h-${now.getMinutes()}m-${now.getSeconds()}s`
    return `sleep_recording_${formattedDate}.mp4`
  }

  const autoDelete = () => {
    const now = new Date()
    const currentYear = now.getFullYear()
    const currentMonth = now.getMonth() + 1
    const currentDay = now.getDate()
    for (let i = 0; i < audioFiles.length; i++) {
      const filename = audioFiles[i].filename
      const month = parseInt(filename.slice(16, 18))
      const day = parseInt(filename.slice(19, 21))
      const year = parseInt(filename.slice(22, 26))
      if (year === currentYear && month === currentMonth) {
        if (currentDay - day <= expiryThreshold) {
          handleDelete(audioFiles[i].file_id)
          console.log(`File ${filename} auto deleted`)
        }
      }
    }
  }

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        await getAccessTokenSilently({
          audience: audience,
          ignoreCache: true,
        });
      } catch (error) {
        console.error('Error refreshing token', error);
      }
    }, 10 * 60 * 1000); // Refresh every 10 minutes

    return () => clearInterval(interval);
  }, [getAccessTokenSilently]);

  const releaseWakeLock = async () => {
    if (wakeLock !== null) {
      await wakeLock.release();
      setWakeLock(null)
      console.log("wakelock released")
    }
  }

  const fetchAudioFiles = async () => {
    if (!isAuthenticated || !user) {
      return;
    }
    try {
        const token = await getAccessTokenSilently({
           audience: audience,
        });
        setToken(token)
        const response = await axios.get(`${apiUrl}/audio-files`, {
            headers: {Authorization: `Bearer ${token}`}
        })
        setAudioFiles(response.data)
    } catch (error) {
      console.error("Error fetching files", error)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true}) // asks browser for mic permissions, audio: true specifies only audio is recorded
      mediaStreamRef.current = stream;

      if (!mediaStreamRef.current) {
        console.error("No media stream available.");
        return;
      }
      recorderRef.current = RecordRTC(mediaStreamRef.current, {
        type:'audio',
        mimeType: 'audio/mp4;codecs=mp4a.40.2',
        timeSlice: 5000,
        desiredSampRate: 44100, 
        numberOfAudioChannels: 1,
        bufferSize: 4096,
        recorderType: RecordRTC.StereoAudioRecorder,
        ondataavailable: handleData, // callback to handle each blob data
      })
      recorderRef.current.startRecording();

      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone', error)
    }
    await requestWakeLock();
  }
  
  const handleData = (blob) => {
    processAudioChunk(blob);
  };

  const processAudioChunk = (audioBlob) => {
    setTimeout(async () => {
      const rms = await calculateRMS(audioBlob);
      setEnergyLog((prev) => [...prev, rms])
      try {
        console.log("RMS ENERGY: ", rms);
        if (rms > energyThreshold) {
          await uploadAudioChunk(audioBlob);
          console.log('Audio Uploaded');
        } else {
          console.log('Audio Discarded due to low energy');
        }
      } catch (err) {
        console.error("Error processing chunk", err)
      }
    }, 0);
  };

  const calculateRMS = (blob) => {
    return new Promise((resolve, reject) => {
      const audioContext = new window.AudioContext();
      const reader = new FileReader();
      reader.onload = (event) => {
        audioContext.decodeAudioData(event.target.result)
          .then((audioBuffer) => {
            const channelData = audioBuffer.getChannelData(0);
            let sum = 0;
            for (let i = 0; i < channelData.length; i++) {
              sum += channelData[i] * channelData[i];
            }
            const rms = Math.sqrt(sum / channelData.length);
            audioContext.close();
            resolve(rms);
          })
          .catch((error) => {
            console.error('Error decoding audio data', error);
            audioContext.close();
            reject(error);
          });
      }
      reader.onerror = (error) => {
        console.error("error reading blob", error)
        reject(error)
      }
      reader.readAsArrayBuffer(blob)
    })
  }

  const stopRecording = () => {
    if (!isRecordingRef.current) return;

    recorderRef.current.stopRecording();
    setIsRecording(false)
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    releaseWakeLock();
  }

  const uploadAudioChunk = async (audioData) => {
    try {
      const token = await getAccessTokenSilently({
        audience: audience,
      });
      const formData = new FormData() // holds audio file data: name, audio, and eventually the id
      const fileName = getFileName() // generate unique dated name
      formData.append('file', audioData, fileName) // add necessary info to the object formData

      const response = await axios.post(`${apiUrl}/upload-audio`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`
        }, // send post request with body of formData and header that defines the Content-Type
      })
      console.log("Audio uploaded successfully", response.data)
    } catch (error) {
      console.error("Upload failed: ", error)
    }
    fetchAudioFiles();
  }

  const handleDelete = async (file_id) => {
    try {
      const token = await getAccessTokenSilently({
        audience: audience,
      });
      const response = await axios.delete(`${apiUrl}/audio-file/${file_id}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      })
      console.log('Deleted Audio File')
      fetchAudioFiles();
    } catch (error) {
      console.log("Error deleting", error)
    }
  }

  const clearLog = () => {
    setEnergyLog([])
  }

  const updateEnergyThreshold = () => {
    const sum = energyLog.reduce((a, b) => a + b);
    const avg = (sum / energyLog.length).toFixed(5)
    setEnergyThreshold(avg)
  }

  return (
  <div className="container">
    <div className="header">
        <Link to={"/user-guide"} >
          <div className="about">
            <button className='manual-button'>User Guide</button>
          </div>
        </Link>

      <div className="auth-buttons">
        <LoginButton className="login-button" isAuthenticated={isAuthenticated}/>
        <LogoutButton className="logout-button" />
      </div>
    </div>

  {!isAuthenticated && (
    <div className="login-prompt">Please login to use Snooze Scribe</div>
  )}

  {isAuthenticated && (
    <div className="recorder-section">
      <h2 className="section-title">Snooze Scribe</h2>

      <div className="threshold-control">
        <label htmlFor="energyThreshold">Energy Threshold:</label>
        <input
          type="number"
          id="energyThreshold"
          placeholder="Energy Threshold"
          value={energyThreshold}
          onChange={(e) => setEnergyThreshold(parseFloat(e.target.value))}
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

      {isRecording && <div className="warning">Recording... Cannot play while recording active</div>}

      <div className="audio-files-section">
        <h3 className="section-subtitle">Audio Files</h3>
        <ul className="audio-files-list">
          {audioFiles.map((file) => (
            <li key={file.file_id} className="audio-file-item">
              <span className="file-name">{file.filename}</span>
              <div className="file-actions">
                <Waveform
                  audioUrl={`${apiUrl}/${file.audio_url}`}
                  token={token}
                />
                <button onClick={() => handleDelete(file.file_id)} className="delete-button">
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )}

  {isAuthenticated && (
    <div className="layer">
      <div className="energy-log">
        <h2 className="section-title">Energy Log</h2>
        <ul className='energy-level-list'>
          {energyLog.map((energy, index) => (
            <li key={index} className='energy-level-item'>
              Energy Level: {energy}
            </li>
          ))}
        </ul>
        <div className="log-buttons">
          <button className="clear-log" onClick={clearLog}>Clear Log</button>
          <button className="set-threshold" onClick={updateEnergyThreshold} disabled={energyLog.length <= 5}>Set Threshold</button>
        </div>
      </div>
      <div className="calibrate-manual">
        <h2 className="section-title">Calibration Instructions</h2>
        <ol className="calibrate-instructions">
          <li>Ensure an environment with minimal noise</li>
          <li>Press the "Start Recording" Button</li>
          <li>Again, ensure idle room noise is the only active noise</li>
          <li>Allow for the Energy Log to populate with entries</li>
          <li>Ensure the entries are accurate and unaffected by external noise, if not clear the log</li>
          <li>Press the "Set Threshold" button at 5 entries or greater</li>
          <li>Alternatively, set the threshold manually based on observations of the energy log</li>
        </ol>
      </div>
    </div>
  )}
</div>

  )
}

export default AudioRecorder;

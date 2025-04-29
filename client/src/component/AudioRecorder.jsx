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
  const [expiryThreshold, setExpiryThreshold] = useState(2);
  const [maxAudioFiles, setMaxAudioFiles] = useState(500)
  const [token, setToken] = useState(null);
  const [energyLog, setEnergyLog] = useState([]);
  const [wakeLock, setWakeLock] = useState(null); // keeps screen on during the night for IOS Mobile Users because Webkit API is terrible and halts all background processes past a grace period on sleep
  const [classifying, setClassifying] = useState(null); // Track which file is being classified
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
    autoDelete();
    const interval = setInterval(autoDelete, 120 * 60 * 1000); // Every 60 minutes
    return () => clearInterval(interval);
  }, [audioFiles]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording])

  useEffect(() => {
    stopRecording()
  }, [])

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
    const now = new Date();
    
    audioFiles.forEach(file => {
      const filename = file.filename; // Expected format: 'sleep_recording_MM-DD-YYYY_hh-mm-ss.mp4'
      
      const regex = /sleep_recording_(\d{1,2})-(\d{1,2})-(\d{4})_/;
      const match = filename.match(regex);
      
      if (match) {
        const month = parseInt(match[1], 10);
        const day = parseInt(match[2], 10);
        const year = parseInt(match[3], 10);
        
        const fileDate = new Date(year, month - 1, day);
        const diffTime = now - fileDate;
        const diffDays = diffTime / (1000 * 60 * 60 * 24);
        
        if (diffDays >= expiryThreshold) {
          handleDelete(file.file_id);
          console.log(`File ${filename} auto deleted (Age: ${diffDays.toFixed(2)} days)`);
        }
      } else {
        console.warn(`Filename "${filename}" does not match the expected format.`);
      }
    });
  };
  

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
        timeSlice: 15000,
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
      setEnergyLog((prev) => {
        if (prev.length <= 50) {
          return [...prev, rms]
        }
        return prev
      })
      try {
        console.log("RMS ENERGY: ", rms);
        if (rms > energyThreshold) {
          setAudioFiles((prev) => {
            if (prev.length < maxAudioFiles) {
              uploadAudioChunk(audioBlob);
              console.log("Uploading")
            } else {
              console.log("Maximum files reached")
            }
            return prev
          })
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
      console.log('Deleted Audio File!!')
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

  const handleClassify = async (fileId) => {
    try {
      setClassifying(fileId);
      const token = await getAccessTokenSilently({
        audience: audience,
      });
      
      const response = await axios.post(`${apiUrl}/classify-audio/${fileId}`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Update the file's classification in the local state
      setAudioFiles(prevFiles => 
        prevFiles.map(file => 
          file.file_id === fileId 
            ? { 
                ...file, 
                classification: response.data.classification,
                confidence: response.data.confidence
              } 
            : file
        )
      );
      
      console.log(`File ${fileId} classified as ${response.data.classification} with confidence ${response.data.confidence}`);
    } catch (error) {
      console.error("Error classifying file:", error);
    } finally {
      setClassifying(null);
    }
  };

  return (
  <div className="container">
    <div className="header">
        <Link to={"/user-guide"} >
          <div className="about">
            <button className='manual-button' onClick={stopRecording}>User Guide</button>
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
          value={energyThreshold}
          placeholder="Energy Threshold"
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

      <div className="audio-files-section">
        <h3 className="section-subtitle">Audio Files</h3>
        <ul className="audio-files-list">
          {audioFiles.map((file) => (
            <li key={file.file_id} className="audio-file-item">
              <div className="file-header">
                <span className="file-name">{file.filename}</span>
                {file.classification && file.classification !== "unclassified" && (
                  <span className={`classification-badge ${file.classification}`}>
                    Class: {file.classification} ({(file.confidence * 100).toFixed(0)}%)
                  </span>
                )}
              </div>
              <div className="file-actions">
                <Waveform
                  audioUrl={`${apiUrl}/${file.audio_url}`}
                  token={token}
                />
                <div className="file-buttons">
                  <button 
                    onClick={() => handleClassify(file.file_id)} 
                    className="classify-button"
                    disabled={classifying === file.file_id}
                  >
                    {classifying === file.file_id ? 'Classifying...' : 'Classify'}
                  </button>
                  <button 
                    onClick={() => handleDelete(file.file_id)} 
                    className="delete-button"
                  >
                    Delete
                  </button>
                </div>
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

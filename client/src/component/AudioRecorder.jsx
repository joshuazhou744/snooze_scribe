import { useState, useRef, useEffect } from 'react'
import { useAuth0 } from '@auth0/auth0-react';
import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';
import axios from 'axios'
import './AudioRecorder.css';

const AudioRecorder = () => {
  const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();
  const [audioFiles, setAudioFiles] = useState([]);
  const [isRecording, setIsRecording] = useState(false)
  const isRecordingRef = useRef(isRecording);
  const [energyThreshold, setEnergyThreshold] = useState(0.007)
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const audioPlayerRef = useRef(null)
  const apiUrl = import.meta.env.VITE_API_URL;
  const audience= import.meta.env.VITE_AUTH0_AUDIENCE;

  useEffect(() => {
    fetchAudioFiles()
    const interval = setInterval(fetchAudioFiles, 20000)
    return () => clearInterval(interval)
  }, [isAuthenticated])

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording])

  const fetchAudioFiles = async () => {
    if (!isAuthenticated || !user) {
      return;
    }
    try {
        const token = await getAccessTokenSilently({
            audience: audience,
        });
        const response = await axios.get(`${apiUrl}/audio-files`, {
            headers: {Authorization: `Bearer ${token}`}
        })
        setAudioFiles(response.data)
    } catch (error) {
      console.error("Error fetching files", error)
    }
  }

  const getFileName = () => {
    const now = new Date()
    const formattedDate = `${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}_${now.getHours()}h-${now.getMinutes()}m-${now.getSeconds()}s`
    return `sleep_recording_${formattedDate}.mp4`
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true}) // asks browser for mic permissions, audio: true specifies only audio is recorded
      mediaStreamRef.current = stream;

      audioContextRef.current = new AudioContext() // creates an AudioContext() that represents the Web Audio API, allows audio manipulation
      const source = audioContextRef.current.createMediaStreamSource(stream) // converts audio stream from mic to something the AudioContext() can process, source is the audio source node created from mic stream
      const analyser = audioContextRef.current.createAnalyser() // creates Analyser node that can read and analyze audio in real time
      analyser.fftSize = 4096; // fft = fast fourier transform, size 2048, breaks audio into frequency components
     
      source.connect(analyser) // connect audio stream to analyser
      analyserRef.current = analyser;

      setIsRecording(true)

      startMediaRecorder()
    } catch (error) {
      console.error('Error accessing microphone', error)
    }
  }
  
  const startMediaRecorder = () => {
    if (!mediaStreamRef.current) {
      console.error("No media stream available.");
      return;
    }
    mediaRecorderRef.current = new MediaRecorder(mediaStreamRef.current, { mimeType: 'audio/webm; codecs=opus' });

    mediaRecorderRef.current.ondataavailable = handleData;

    mediaRecorderRef.current.onstop = () => {
      if (isRecordingRef.current) {
        // Start a new MediaRecorder instance for the next chunk
        startMediaRecorder();
      } else {
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
      }
    }
    mediaRecorderRef.current.start(); // Start recording

  // Schedule the recorder to stop after 5 seconds
    setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    }, 5000);
  }
  
  const handleData = (event) => {
    if (event.data.size > 0) {
      processAudioChunk(event.data);
    }
  };

  const processAudioChunk = (audioBlob) => {
    setTimeout(async () => {
      const rms = await calculateRMS(analyserRef.current);
      console.log("RMS ENERGY: ", rms);
      if (rms > energyThreshold) {
        await uploadAudioChunk(audioBlob);
        console.log('Audio Uploaded');
      } else {
        console.log('Audio Discarded due to low energy');
      }
    }, 0);
  };

  const calculateRMS = (analyser) => {
    return new Promise((resolve) => {
      const bufferLength = analyser.fftSize;
      const dataArray = new Uint8Array(bufferLength)

      analyser.getByteTimeDomainData(dataArray);

      let sum = 0
      for (let i = 0; i < bufferLength; i++) {
        const normalized = dataArray[i] / 128 - 1;
        sum += normalized ** 2;
      }
      const rms = Math.sqrt(sum / bufferLength);
      resolve(rms);
    })
  }

  const stopRecording = () => {
    setIsRecording(false)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }

  const uploadAudioChunk = async (audioData) => {
    try {
      const token = await getAccessTokenSilently();
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

  const playAudio = async (audioUrl) => {
    if (audioPlayerRef.current) {
      try {
        const token = await getAccessTokenSilently({
          audience: audience,
        });
        const response = await axios.get(audioUrl, {
          responseType: 'blob',
          headers: {
            Authorization: `Bearer ${token}`
          }
        })
        const blob = response.data;
        const url = URL.createObjectURL(blob)
        audioPlayerRef.current.src = url;
        const playPromise = audioPlayerRef.current.play();
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              console.log('playing audio')
            })
            .catch(error => {
              console.error("playback failed", error)
            })
        }
        audioPlayerRef.current.onended = () => {
          URL.revokeObjectURL(url)
        }
      
      } catch (error) {
        console.error('Error fetching audio', error);
      }
    }
  };

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
      console.log(response.data)
      fetchAudioFiles();
    } catch (error) {
      console.log("Error deleting", error)
    }
  }

  return (
    <div className="container">
  <div className="auth-buttons">
    <LoginButton className="login-button" />
    <LogoutButton className="logout-button" />
  </div>

  {!isAuthenticated && (
    <div className="login-prompt">Please log in to record and view audio files</div>
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
          step="0.1"
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
                <button
                  onClick={() => playAudio(`${apiUrl}/audio-file/play/${file.file_id}`)}
                  disabled={isRecording}
                  className="play-button"
                >
                  Play
                </button>
                <button onClick={() => handleDelete(file.file_id)} className="delete-button">
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <audio ref={audioPlayerRef} controls className="audio-player">
        Your browser does not support the audio element.
      </audio>
    </div>
  )}
</div>

  )
}

export default AudioRecorder;

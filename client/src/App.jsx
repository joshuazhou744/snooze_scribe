import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const AudioRecorder = () => {
  const [audioFiles, setAudioFiles] = useState([]);
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const analyserRef = useRef(null)
  const audioContextRef = useRef(null)
  const energyThreshold = 0.0067;
  const apiUrl = "http://localhost:8000"

  useEffect(() => {
    const fetchAudioFiles = async () => {
      try {
        const response = await axios.get(`${apiUrl}/audio-files`)
        setAudioFiles(response.data)
      } catch (error) {
        console.error("Error fetching files", error)
      }
    }
    fetchAudioFiles()
    const interval = setInterval(fetchAudioFiles, 15000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (isRecording) { // calls startRecording when isRecording is true
      startRecording();
    }
    return stopRecording // recording stops when isRecording becomes false
  }, [isRecording]) // useEffect is executed whenever isRecording changes

  const getFileName = () => {
    const now = new Date()
    const formattedDate = `${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}_${now.getHours()}h-${now.getMinutes()}m-${now.getSeconds()}s`
    return `sleep_recording_${formattedDate}.wav`
  }

  const startRecording = async () => {
    setIsRecording(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true}) // asks browser for mic permissions, audio: true specifies only audio is recorded
      audioContextRef.current = new AudioContext() // creates an AudioContext() that represents the Web Audio API, allows audio manipulation
      const source = audioContextRef.current.createMediaStreamSource(stream) // converts audio stream from mic to something the AudioContext() can process, source is the audio source node created from mic stream

      analyserRef.current = audioContextRef.current.createAnalyser() // creates Analyser node that can read and analyze audio in real time
      analyserRef.current.fftSize = 2048; // fft = fast fourier transform, size 2048, breaks audio into frequency components
      source.connect(analyserRef.current) // connect audio stream to analyzer for processing

      mediaRecorderRef.current = new MediaRecorder(stream) // MediaRecorder captures and stores audio in chunks
      mediaRecorderRef.current.ondataavailable = handleDataAvailable // trigger handleDataAvailable every 5 seconds
      mediaRecorderRef.current.start(5000); // breaks recording in 5 second intervals
    } catch (error) {
      console.error('Error accessing microphone', error)
    }
  }

  const handleDataAvailable = async (e) => {
    const audioData = e.data
    if (audioData.size > 0) {
      const isSignificant = checkEnergyLevel()

      if (isSignificant) {
        uploadAudioChunk(audioData)
      } else {
        console.log('Discarding Chunk due to low energy')
      }
    }
  }

  const checkEnergyLevel = () => {
    const bufferLength = analyserRef.current.fftSize; // gets the time window of the recorded data, can use this to get average energy in the 5s interval, higher fft window means more detailed processing
    const dataArray = new Uint8Array(bufferLength) // creates new array of 8-bit unassigned numbers
    analyserRef.current.getByteTimeDomainData(dataArray) // fills dataArray with the current time domain from the AnalyserNode reference

    let sumSquares = 0;
    for (let i = 0; i < bufferLength; i++) {
      const normalized = dataArray[i] / 128 - 1 // each value in dataArray is between 0-255 so dividing by 128 then subtracting 1 gives a value between -1 and 1, normalizing the data for easier processing, amplitude is often represented from -1 to 1 so here we display the 0-255 volume value as amplitude from -1 to 1
      // normalizing also allows accurately reflected RMS levels in this scenario
      sumSquares += normalized * normalized // add the squared value to the sumSquares array, always will be positive
    }
    const rms = Math.sqrt(sumSquares / bufferLength) // calculate rms energy level

    return rms > energyThreshold
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
  }

  const uploadAudioChunk = async (audioData) => {
    const formData = new FormData() // holds audio file data: name, audio, and eventually the id
    const fileName = getFileName() // generate unique dated name
    formData.append('file', audioData, fileName) // add necessary info to the object formData

    try {
      const response = await axios.post(`${apiUrl}/upload-audio`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }, // send post request with body of formData and header that defines the Content-Type
      })
      console.log("Audio uploaded successfully", response.data)
    } catch (error) {
      console.error("Upload failed: ", error)
    }
  }

  const playAudio = async (fileId) => {
    try {
      const response = await axios.get(`${apiUrl}/audio-file/play/${fileId}`, {
        responseType: 'blob', // ensures that the audio is returned as binary data (blob)
      });
      const audioUrl = URL.createObjectURL(new Blob([response.data], { type: 'audio/wav' }));
      const audio = new Audio(audioUrl); // Create an HTML audio element
      audio.play(); // Play the audio file
    } catch (error) {
      console.error("Error playing audio", error);
    }
  };

  return (
    <div>
      <h2>Audio Recorder with Energy Detection</h2>
      <button onClick={() => setIsRecording(true)} disabled={isRecording}>
        Start Recording
      </button>
      <button onClick={stopRecording} disabled={!isRecording}>
        Stop Recording
      </button>
      {isRecording && <div className='warn'>Cannot play while recording active</div>}
      
      <h3>Audio Files</h3>
      <ul>
        {audioFiles.map((file) => (
          <li key={file.file_id}>
            {file.filename}
            <button onClick={() => playAudio(file.file_id)} disabled={isRecording}>Play</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default AudioRecorder

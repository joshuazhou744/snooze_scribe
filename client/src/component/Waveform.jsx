import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { useWavesurfer } from '@wavesurfer/react'
import Timeline from 'wavesurfer.js/dist/plugins/timeline.esm.js'
import axios from 'axios';

export default function Waveform({audioUrl, token}) {
  const waveformRef = useRef(null)
  const [blob, setBlob] = useState(null)

  useEffect(() => {
    const fetchAudio = async () => {
      try {
        const response = await axios.get(audioUrl, {
          responseType: 'blob',
          headers: {
            Authorization: `Bearer ${token}`
          }
        })

        const audioBlob = response.data
        const url = URL.createObjectURL(audioBlob)
        setBlob(url)
      } catch (err) {
        console.error("Error fetching blob: ", err)
      }
    }
    if (audioUrl && token) {
      fetchAudio();
    }
    return () => {
      if (blob) {
        URL.revokeObjectURL(blob)
        setBlob(null)
      }
    }
  }, [audioUrl, token])

  const formatTime = (seconds) => [seconds / 60, seconds % 60].map((v) => `0${Math.floor(v)}`.slice(-2)).join(':')

  const { wavesurfer, isPlaying, currentTime } = useWavesurfer({
    container: waveformRef,
    height: 80,
    waveColor: 'violet',
    progressColor: 'purple',
    cursorColor: 'navy',
    url: blob,
    plugins: useMemo(() => [Timeline.create()], []),
  })

  const onPlayPause = useCallback(() => {
    wavesurfer && wavesurfer.playPause()
  }, [wavesurfer])

  return (
    <>
      <div ref={waveformRef}/>

      <p>Current time: {formatTime(currentTime)}</p>

      <button onClick={onPlayPause} style={{ minWidth: '5em' }} className='play-button'>
          {isPlaying ? 'Pause' : 'Play'}
      </button>
    </>
  )
}

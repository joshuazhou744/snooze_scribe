# Snooze Scribe

Free sleep recorder so I don't have to pay for one on the app store

## App Link

https://snooze-scribe.vercel.app

## Versions and Updates:

v1 (10/07/2024): 
- Deployed on Vercel
- Saves audio as .webm
- MediaRecorder API is not compatible with mobile browsers; cannot record on mobile
- HTML5 audio element is not supported with mobile browsers; cannot playback on mobile

v2 (10/07/2024): 
- Overhauled MediaRecorder API to use RecordRTC as the web recorder; now records in .mp4 format
- Able to record on mobile browsers; still cannot play recorded audio, fix soon

v3 (10/08/2024): 
- Implemented WaveSurfer.js and its Timeline plugin as audio player and waveform visualizer with user interface
- Mobile browsers can record but cannot see the visualizer nor play the recorded audio, you will have to access your recordings on a computer browser

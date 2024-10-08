# Snooze Scribe

Free sleep eecorder so I don't have to pay for one on the app store

## App Link

https://snooze-scribe.vercel.app/

## Versions and Updates:

v1 (10/07/2024): Deployed on Vercel; saves audio as .webm; MediaRecorder API is not compatible with mobile browsers; HTML5 audio element is not supported with mobile browsers

v2 (10/07/2024): Overhauled MediaRecorder API to use RecordRTC as the web recorder; now records in .mp4 format; can record on mobile browsers; still cannot play recorded audio, fix soon

v3 (In Progress): Implementing WaveSurfer.js as audio player with user interface to see waveform patterns and allow user to choose playback segments; adding energy level log so user can calibrate the energy threshold according to the room noise

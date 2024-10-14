# Snooze Scribe

Free sleep recorder so I don't have to pay for one on the app store

## App Link

https://snooze-scribe.vercel.app

## Versions and Updates:

v0 (10/02/2024):
- Snooze Scribe run on local development server with audio recording and handling on the python backend
- Audio files are saved locally in an "audio" folder in the project directory
- Legacy code still exists (commented out) in main.py of the api if you want to run this locally
- Uses PyAudio SoundDevice and SoundFile to record and procesgs audio data, possibly more reliable than RecordRTC

v1.0 (10/07/2024): 
- Deployed on Vercel
- Saves audio as .webm
- MediaRecorder API is not compatible with mobile browsers; cannot record on mobile
- HTML5 audio element is not supported with mobile browsers; cannot playback on mobile

v1.1 (10/07/2024): 
- Overhauled MediaRecorder API to use RecordRTC as the web recorder; now records in .mp4 format
- Able to record on mobile browsers; still cannot play recorded audio, fix soon

v2.0 (10/08/2024): 
- Implemented WaveSurfer.js and its Timeline plugin as audio player and waveform visualizer with user interface
- Mobile browsers can record but cannot see the visualizer nor play the recorded audio, you will have to access your recordings on a computer browser

v2.1 (10/10/2024):
- Created RMS Energy Level log to help user calibrate the energy threshold accordingly to room noise, included calibration instructions
- Added the WakeLock API to ensure the recorder stays active throughout the night 
- IOS automatically shuts down all background processes after a grace period during sleep; this function keeps the recording active at night; the phone will have to be charging during the recording process

v2.11 (10/10/2024):
- Finished user guide and additional informations sections
- Added responsive CSS styles for mobile devices

v2.2 (10/14/2024):
- Added auto delete function for audio files past 2 days
- Increased clipping interval and implemented a maximum audio file list to prevent database overflow

v3 (In Progress):
- Building a "tips" section that allows users to try methods to reduce snoring and informs users on things that expedite snoring
- Start working on a "Snooze Score" value that reflects the user's quality of sleep based on the amount snoring detected using frequency detection from Web Audio API
    - This value will also be affected by the snoring trigger factors (alcohol, smoking, etc.) and the methods that reduce snoring (nasal strips, mouth tape)
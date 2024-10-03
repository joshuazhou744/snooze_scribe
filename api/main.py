from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from fastapi.responses import StreamingResponse

import soundfile as sf
import sounddevice as sd
import numpy as np
from datetime import datetime as dt
from bson import ObjectId
from gridfs.errors import NoFile

from dotenv import load_dotenv
import os
import io
import asyncio

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

### VARIABLES

MONGODB_URL = os.getenv("MONGODB_URL")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client["audio"]
fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db, bucket_name="sleep_recorder") # bucket is the storage unit used by gridfs, it's an interface with the data gridfs stores, bucket name makes it so the .files and .chunks collections are created in the collection sleep_recorder
# bucket has 2 collections: fs.files and fs.chunks
# when you upload, the file is split into small chunks and stored in fs.chunks, the reference and metadata information is stored in fs.files
# when you retrieve, the file is reassembled by reading the chunks in sequence

duration = 5 # 5 second recording chunks
sample_rate = 44100  # samples of a waveform per second to create an accurate signal
energy_threshold = 0.01  # volume threshold to record
recording_active = False

audio_files = []

### HELPER FUNCTIONS

async def detect_sound():
    global recording_active
    while recording_active:
        print("listening")

        audio = sd.rec(int(duration*sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        if np.isnan(audio).any():
                print("Warning: audio contains NaN values, skipping detection")
                continue
        
        rms_energy = np.sqrt(np.mean(audio**2)) # root mean squared value is avg sound energy
        if rms_energy > energy_threshold:
             print(f"Sound Detected: Energy: {rms_energy}")
             file_name = get_file_name()
             await save_file_to_server(audio, file_name)
        else:
             print(f"No audio detected: Energy: {rms_energy}")
        await asyncio.sleep(0.5)


def get_file_name():
    now = dt.now().strftime("%m-%d_%Hh-%Mm-%Ss")
    file_name = f"sleep_recording_{now}.wav"
    return file_name

async def save_file_to_server(audio, file_name):
     audio_buffer = io.BytesIO() # creates an in-memory binary stream to hold audio data
     sf.write(audio_buffer, audio, sample_rate, format="WAV")

     audio_buffer.seek(0) # resets in-memory buffer position to 0 so a new audio data can be read at position 0
     await fs.upload_from_stream(file_name, audio_buffer) # uploads the audio data in the buffer to mongodb gridfs under file_name


### API CALLS

@app.post("/start-recording")
async def start_recording():
    global recording_active
    if recording_active:
        return {"message": "Recording is already active"}
    recording_active = True
    asyncio.create_task(detect_sound()) # rather than await detect_sound(), allows detect_sound() to run concurrently with api calls
    return {"message": "Recording started"}

@app.post("/stop-recording")
async def stop_recording():
    global recording_active
    if not recording_active:
        return {"message": "Recording is already inactive"}
    recording_active = False
    return {"message": "Recording stopped"}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)): # UploadFile is a class used to handle file uploads, File(...) the input will be a file and the file input is required as denoted by ...
     try:
        audio_data = await file.read() # all raw binary data is stored into audio_data

        audio_buffer = io.BytesIO(audio_data) # the in-memory binary stream contains the audio binary data
        file_id = await fs.upload_from_stream(file.filename, audio_buffer) # uploads the name and audio binary data, returns the generated id for the stored file
        return {"message": "Audio uploaded successfully", "_id": str(file_id)}
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Error uploading data {e}")


@app.get("/audio-files")
async def get_audio_files():
    cursor = fs.find() # queries all files stored in the gridfs bucket, cursor is an iterable object that fetches documents
    files = await cursor.to_list(None) # converts cursor to a python list, None parameter means there is no limit to the list; retrieves all files
    file_list = []
    for file in files:
         file_data = {
            "file_id": str(file['_id']), # _id is mongodb's ObjectId
            "filename": file['filename'] # get the filename stored in mongodb
         }
         file_list.append(file_data)
    return file_list

@app.get("/audio-file/play/{file_id}")
async def play_audio(file_id: str):
     try:
          grid_out = await fs.open_download_stream(ObjectId(file_id)) # retrieve audio file by ObjectId
          return StreamingResponse(grid_out, media_type="audio/wav") # stream audio response
     except Exception as e:
          raise HTTPException(status_code=404, detail=f"File not found: {e}")

@app.delete("/audio-file/{file_id}")
async def delete_audio_file(file_id: str):
     try: 
        await fs.delete(ObjectId(file_id)) # ObjectId converts the file_id string to a bson.ObjectId which is what mongodb reads
        return {"message":"File deleted"}
     except NoFile:
          raise HTTPException(status_code=404, detail="File not found")
     except Exception as e:
          raise HTTPException(status_code=404, detail=f"An error occurred: {e}")
     
@app.on_event("shutdown")
async def shutdown_event():
    global recording_active
    recording_active = False
    print("SHUTTING DOWN")
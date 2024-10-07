from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from fastapi.responses import StreamingResponse

import soundfile as sf
import sounddevice as sd
import numpy as np
from datetime import datetime as dt
from bson import ObjectId
from gridfs.errors import NoFile
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import io
import base64
import asyncio
import logging

from jose import jwt, JWTError
from typing import List
import requests
import hashlib


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://snooze-scribe.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### VARIABLES


AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN')
MONGODB_URL = os.getenv("MONGODB_URL")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client["audio"]
fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db, bucket_name="sleep_recorder") # bucket is the storage unit used by gridfs, it's an interface with the data gridfs stores, bucket name makes it so the .files and .chunks collections are created in the collection sleep_recorder
# bucket has 2 collections: fs.files and fs.chunks
# when you upload, the file is split into small chunks and stored in fs.chunks, the reference and metadata information is stored in fs.files
# when you retrieve, the file is reassembled by reading the chunks in sequence

"""duration = 5 # 5 second recording chunks
sample_rate = 44100  # samples of a waveform per second to create an accurate signal
energy_threshold = 0.01  # volume threshold to record
recording_active = False

audio_files = []"""

### HELPER FUNCTIONS
"""
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
    file_name = f"sleep_recording_{now}.mp3"
    return file_name

async def save_file_to_server(audio, file_name):
     audio_buffer = io.BytesIO() # creates an in-memory binary stream to hold audio data
     sf.write(audio_buffer, audio, sample_rate, format="WAV")

     audio_buffer.seek(0) # resets in-memory buffer position to 0 so a new audio data can be read at position 0
     await fs.upload_from_stream(file_name, audio_buffer) # uploads the audio data in the buffer to mongodb gridfs under file_name
"""

"""def encode_audio_to_base64(audio_data):
    return base64.b64encode(audio_data).decode('utf-8')"""

def get_public_key():
     jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
     jwks = requests.get(jwks_url).json()
     return jwks['keys'][0]


def get_auth_token(authorization: str = Header(None)):
     if authorization is None:
          raise HTTPException(status_code=403, detail="No authorization header")
     try:
          token = authorization.split(" ")[1]
          return token
     except IndexError:
          raise HTTPException(status_code=403, detail="Invalid authorization header")
     
def verify_jwt_token(token: str): # verify the user by getting it's 'sub' or id
     try:
          jwks_key = get_public_key()
          rsa_key = {
               'kty': jwks_key['kty'],
               'kid': jwks_key['kid'],
               'use': jwks_key['use'],
               'n': jwks_key['n'],
               'e': jwks_key['e']
          }
          payload = jwt.decode(
               token, 
               rsa_key, 
               algorithms=["RS256"],
               audience=AUTH0_AUDIENCE,
               issuer=f"https://{AUTH0_DOMAIN}/"

          )
          return payload['sub'] # reload user id
     except JWTError as e:
          print("TOKEN BAD")
          raise HTTPException(status_code=403, detail=f"Token invalid or expired {e}")
     
def get_user_gridfs(user_id: str):
     hashed_user_id = hashlib.sha256(user_id.encode('utf-8')).hexdigest()

     db = client['audio']
     user_collection_name = f"{hashed_user_id}"
     gridfs_files = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db, bucket_name=user_collection_name)
     return gridfs_files

### API CALLS

"""@app.post("/start-recording")
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
    return {"message": "Recording stopped"}"""


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), authorization: str = Header(None)): # UploadFile is a class used to handle file uploads, File(...) the input will be a file and the file input is required as denoted by ...
     token = get_auth_token(authorization)
     user_id = verify_jwt_token(token)
    
     gridfs_files = get_user_gridfs(user_id)
     
     try:
        audio_data = await file.read() # all raw binary data is stored into audio_data
        audio_buffer = io.BytesIO(audio_data) # the in-memory binary stream contains the audio binary data
        file_id = await gridfs_files.upload_from_stream(file.filename, audio_buffer) # uploads the name and audio binary data, returns the generated id for the stored file
        return {"message": "Audio uploaded successfully", "_id": str(file_id)}
     except Exception as e:
          print(f"Error occurred while uploading audio: {e}")
          raise HTTPException(status_code=500, detail=f"Error uploading data {e}")
     
@app.get("/download-file")
async def download_file(file_id: str):
     grid_out = await fs.open_download_stream(ObjectId(file_id))
     with open(f"{file_id}_output.mp3", "wb") as f:
          f.write(await grid_out.read())

@app.get("/audio-files")
async def get_audio_files(authorization: str = Header(None)):
    token = get_auth_token(authorization)
    user_id = verify_jwt_token(token)
    
    gridfs_files = get_user_gridfs(user_id)

    cursor = gridfs_files.find() # queries all files stored in the gridfs bucket, cursor is an iterable object that fetches documents
    files = await cursor.to_list(None) # converts cursor to a python list, None parameter means there is no limit to the list; retrieves all files
    file_list = []
    for file in files:
         file_data = {
            "file_id": str(file['_id']), # _id is mongodb's ObjectId
            "filename": file['filename'], # get the filename stored in mongodb
            "audio_url": f"/audio-file/play/{file['_id']}",
         }
         file_list.append(file_data)
    return file_list

@app.get("/audio-file/play/{file_id}")
async def play_audio(file_id: str, authorization: str = Header(None)):
     token = get_auth_token(authorization)
     user_id = verify_jwt_token(token)
     
     gridfs_files = get_user_gridfs(user_id)
     try:
          grid_out = await gridfs_files.open_download_stream(ObjectId(file_id)) # retrieve audio file by ObjectId
          print("stream successfully")
          return StreamingResponse(grid_out, media_type="audio/webm; codecs=opus") # stream audio response
     except Exception as e:
          raise HTTPException(status_code=404, detail=f"File not found: {e}")

@app.delete("/audio-file/{file_id}")
async def delete_audio_file(file_id: str, authorization: str = Header(None)):
     token = get_auth_token(authorization)
     user_id = verify_jwt_token(token)
     
     gridfs_files = get_user_gridfs(user_id)
     try: 
        await gridfs_files.delete(ObjectId(file_id)) # ObjectId converts the file_id string to a bson.ObjectId which is what mongodb reads
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
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from fastapi.responses import StreamingResponse

import numpy as np
from datetime import datetime as dt
from bson import ObjectId
from gridfs.errors import NoFile
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import io

from jose import jwt, JWTError
from typing import List, Optional, Union
import requests
import hashlib

import tempfile
import subprocess
import shutil

# Add imports for classification model
import torch
import librosa

# Define the classification response model
class ClassificationResponse(BaseModel):
    file_id: str
    classification: str
    confidence: float

# Define audio file model
class AudioFile(BaseModel):
    file_id: str
    filename: str
    audio_url: str
    classification: str = "unclassified"
    confidence: float = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "file_id": "60d21b4967d0d631236d5f66",
                "filename": "sleep_recording_10-15-2024_02h-34m-56s.mp4",
                "audio_url": "audio-file/play/60d21b4967d0d631236d5f66",
                "classification": "snore",
                "confidence": 0.92
            }
        }

# Define the efficient CNN model architecture (match the one in snooze_model/main.py)
class EfficientAudioClassifier(torch.nn.Module):
    """Lightweight CNN model for audio classification with fewer parameters"""
    def __init__(self, num_classes, n_mels, fixed_length, sample_rate):
        super(EfficientAudioClassifier, self).__init__()
        
        # Calculate input dimensions based on hop_length
        hop_length = 1024
        self.time_dim = int(fixed_length / hop_length) + 1
        
        # Efficient convolutional architecture with depth-wise separable convolutions
        # First: standard convolution for initial feature extraction
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # Depthwise separable convolution for efficiency
        # Depthwise
        self.conv_dw = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.bn_dw = torch.nn.BatchNorm2d(16)
        # Pointwise
        self.conv_pw = torch.nn.Conv2d(16, 32, kernel_size=1)
        self.bn_pw = torch.nn.BatchNorm2d(32)
        
        # Calculate the flattened size based on input dimensions and network structure
        # After 2 max pooling layers (dividing dimensions by 4)
        n_mels_reduced = n_mels // 4
        time_dim_reduced = self.time_dim // 4
        self.flattened_size = 32 * n_mels_reduced * time_dim_reduced
        
        # Smaller fully connected layers
        self.fc1 = torch.nn.Linear(self.flattened_size, 64)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # First conv block
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        
        # Efficient depthwise separable convolution block
        x = torch.nn.functional.relu(self.bn_dw(self.conv_dw(x)))
        x = self.pool(torch.nn.functional.relu(self.bn_pw(self.conv_pw(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Global variables for model and device
classification_model = None
device = None
normalization_params = None  # Store normalization parameters globally

# Load the trained model
def load_classification_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Look in two possible locations for the model file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'audio_classifier.pth')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"ERROR: Model file not found in any of: {possible_paths}")
            return None, device, None  # Return None for normalization
            
        print(f"Found model at: {model_path}")
            
        # Create model with correct parameters for the efficient architecture
        model = EfficientAudioClassifier(
            num_classes=2, 
            n_mels=64,  # The efficient model uses 64 mel bands instead of 128
            fixed_length=44100*15, 
            sample_rate=44100
        )
        
        print(f"Created model instance: {type(model)}")
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Model checkpoint loaded, keys: {checkpoint.keys()}")
            
            if 'model_state_dict' not in checkpoint:
                print(f"ERROR: Invalid checkpoint format, 'model_state_dict' not found. Available keys: {checkpoint.keys()}")
                return None, device, None  # Return None for normalization
            
            # Get normalization parameters from the checkpoint
            norm_params = None
            if 'normalization' in checkpoint:
                norm_params = {
                    'mean': checkpoint['normalization']['mean'],
                    'std': checkpoint['normalization']['std']
                }
                print(f"Loaded normalization parameters: mean={norm_params['mean']}, std={norm_params['std']}")
            else:
                # Fallback to hardcoded values if not in checkpoint
                norm_params = {
                    'mean': -65.2772445678711,
                    'std': 17.052316665649414
                }
                print(f"WARNING: Using hardcoded normalization values: mean={norm_params['mean']}, std={norm_params['std']}")
                
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()  # Set to evaluation mode
            print(f"Successfully loaded classification model from {model_path}")
            return model, device, norm_params
        except Exception as e:
            print(f"ERROR loading model state: {e}")
            import traceback
            print(traceback.format_exc())
            return None, device, None  # Return None for normalization
    except Exception as e:
        print(f"ERROR in load_classification_model: {e}")
        import traceback
        print(traceback.format_exc())
        return None, device, None  # Return None for normalization

# Extract features from audio for classification (updated for efficient model)
def extract_features(audio_data, sample_rate=44100, n_mels=64, fixed_length=44100*15):
    try:
        print(f"Starting feature extraction, audio data size: {len(audio_data)} bytes")
        
        # Convert audio data to numpy array
        print("Loading audio data with librosa")
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate, mono=True)
        print(f"Audio loaded, length: {len(y)}, sample rate: {sr}")
        
        # Ensure fixed length
        if len(y) < fixed_length:
            print(f"Audio length ({len(y)}) less than required ({fixed_length}), padding...")
            padding = fixed_length - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            print(f"Audio length ({len(y)}) longer than required ({fixed_length}), truncating...")
            y = y[:fixed_length]
            
        # Extract mel spectrogram features - IMPORTANT: use hop_length=1024 to match model
        print("Calculating mel spectrogram...")
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=1024  # Must match the model training parameters
        )
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"Log mel spectrogram shape: {log_mel_spec.shape}, min: {np.min(log_mel_spec)}, max: {np.max(log_mel_spec)}")
        
        return log_mel_spec  # Return raw features WITHOUT normalization
    except Exception as e:
        print(f"ERROR in extract_features: {e}")
        import traceback
        print(traceback.format_exc())
        return None

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://snooze-scribe.vercel.app", "http://localhost:5173", "http://127.0.0.1:5173", "https://snooze-scribe-api-production-7c27c458ab69.herokuapp.com"],
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
          raise HTTPException(status_code=403, detail=f"Token invalid or expired {e}")
     except Exception as e:
          raise HTTPException(status_code=403, detail=f"Token verification error: {e}")
     
def get_user_gridfs(user_id: str):
     hashed_user_id = hashlib.sha256(user_id.encode('utf-8')).hexdigest()

     db = client['audio']
     user_collection_name = f"{hashed_user_id}"
     gridfs_files = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db, bucket_name=user_collection_name)
     return gridfs_files

"""def convert_webm_to_mp4(webm_data: bytes) -> str:
     with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm_file:
          webm_path = temp_webm_file.name
          temp_webm_file.write(webm_data)
     with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_mp4_file:
          mp4_path = temp_mp4_file.name

     try:
          ffmpeg_command = [
               'ffmpeg',
               '-y',  # Overwrite output files without asking
               '-i', webm_path,
               '-c:a', 'aac',
               mp4_path
          ]
          subprocess.run(ffmpeg_command, check=True)
     finally:
          if os.path.exists(webm_path):
               os.remove(webm_path)
     return mp4_path"""

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

@app.on_event("startup")
async def startup_event():
    global classification_model, device, normalization_params
    print("Starting application, loading classification model...")
    classification_model, device, normalization_params = load_classification_model()
    if classification_model is None:
        print("WARNING: Classification model failed to load. Classification endpoints will return 500 errors.")
    elif normalization_params is None:
        print("WARNING: Normalization parameters not loaded. Classification may not be accurate.")


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
     dl_fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db, bucket_name="125ff93b9c0a627f88c0bf6c71ba9188e377c20213b1e92f299c7065f995311f")
     grid_out = await dl_fs.open_download_stream(ObjectId(file_id))
     with open(f"{file_id}_output.mp4", "wb") as f:
          f.write(await grid_out.read())

@app.get("/audio-files", response_model=List[AudioFile])
async def get_audio_files(authorization: str = Header(None)):
    token = get_auth_token(authorization)
    user_id = verify_jwt_token(token)
    
    gridfs_files = get_user_gridfs(user_id)

    cursor = gridfs_files.find() # queries all files stored in the gridfs bucket, cursor is an iterable object that fetches documents
    files = await cursor.to_list(None) # converts cursor to a python list, None parameter means there is no limit to the list; retrieves all files
    file_list = []
    for file in files:
         file_data = AudioFile(
            file_id=str(file['_id']),
            filename=file['filename'],
            audio_url=f"audio-file/play/{file['_id']}",
            classification=file.get('classification', "unclassified"),
            confidence=file.get('confidence', 0.0)
         )
         file_list.append(file_data)
    return file_list

@app.get("/audio-file/play/{file_id}")
async def play_audio(file_id: str, authorization: str = Header(None)):
     token = get_auth_token(authorization)
     user_id = verify_jwt_token(token)
     
     gridfs_files = get_user_gridfs(user_id)
     try:
          grid_out = await gridfs_files.open_download_stream(ObjectId(file_id)) # retrieve audio file by ObjectId
          """webm_data = await grid_out.read()
          mp4_path = convert_webm_to_mp4(webm_data)
          def mp4_streamer():
               with open(mp4_path, "rb") as mp4_file:
                    yield from mp4_file
               if os.path.exists(mp4_path):
                    os.remove(mp4_path)"""
          return StreamingResponse(grid_out, media_type="audio/mp4; codecs=mp4a.40.2") # stream audio response
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

@app.delete("/audio-files/all")
async def delete_all_audio_files(authorization: str = Header(None)):
     token = get_auth_token(authorization)
     user_id = verify_jwt_token(token)
     
     gridfs_files = get_user_gridfs(user_id)
     try:
          # Find all files for the user
          cursor = gridfs_files.find()
          files = await cursor.to_list(None)
          
          if not files:
               return {"message": "No files found to delete"}
          
          # Delete each file
          deleted_count = 0
          for file in files:
               try:
                    await gridfs_files.delete(file['_id'])
                    deleted_count += 1
               except Exception as e:
                    print(f"Error deleting file {file['_id']}: {e}")
                    continue
          
          return {"message": f"Successfully deleted {deleted_count} audio files"}
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"An error occurred while deleting files: {e}")
     
@app.on_event("shutdown")
async def shutdown_event():
    global recording_active
    recording_active = False
    print("SHUTTING DOWN")

@app.post("/classify-audio/{file_id}", response_model=ClassificationResponse)
async def classify_audio(file_id: str, authorization: str = Header(None)):
    try:
        print(f"Starting classification for file_id: {file_id}")
        
        if classification_model is None:
            print("ERROR: Classification model not loaded")
            raise HTTPException(status_code=500, detail="Classification model not loaded")
            
        token = get_auth_token(authorization)
        user_id = verify_jwt_token(token)
        print(f"Authenticated user_id: {user_id}")
        
        # Get user's GridFS bucket
        gridfs_files = get_user_gridfs(user_id)
        
        try:
            # Retrieve the audio file
            print("Attempting to retrieve audio file from GridFS")
            grid_out = await gridfs_files.open_download_stream(ObjectId(file_id))
            audio_data = await grid_out.read()
            print(f"Successfully retrieved audio file, size: {len(audio_data)} bytes")
            
            # Extract features
            print("Extracting audio features")
            features = extract_features(audio_data)
            if features is None:
                print("ERROR: Failed to extract features from audio file")
                raise HTTPException(status_code=500, detail="Could not extract features from audio file")
            print(f"Features extracted, shape: {features.shape}")
            
            # Normalize features
            print("Normalizing features")
            if normalization_params is None:
                print("WARNING: Normalization parameters not loaded")
                raise HTTPException(status_code=500, detail="Model normalization parameters not available")
            else:
                print(f"Using normalization parameters: mean={normalization_params['mean']}, std={normalization_params['std']}")
                features = (features - normalization_params['mean']) / normalization_params['std']
            
            # Convert to tensor
            features = torch.tensor(features, dtype=torch.float32)
            print(f"Final tensor shape: {features.shape}")
                
            # Make prediction
            print("Running model prediction")
            with torch.no_grad():
                features = features.to(device)
                outputs = classification_model(features.unsqueeze(0))  # Add batch dimension
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get prediction for class 1 (snore)
                snore_prob = probabilities[0][1].item()
                print(f"Raw snore probability: {snore_prob}")
                
                # Get highest probability class
                _, predicted = torch.max(probabilities, 1)
                pred_class = predicted.item()
                pred_confidence = probabilities[0][pred_class].item()
                print(f"Predicted class: {pred_class}, confidence: {pred_confidence}")
                
            # Map prediction to class name
            class_names = ["other", "snore"]
            classification = class_names[pred_class]
            print(f"Classification result: {classification}")
            
            # Update the file metadata in MongoDB with classification result
            print("Updating MongoDB document with classification result")
            db_collection = client["audio"][f"{hashlib.sha256(user_id.encode('utf-8')).hexdigest()}.files"]
            await db_collection.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"classification": classification, "confidence": float(pred_confidence)}}
            )
            print("MongoDB document updated successfully")
            
            return {
                "file_id": file_id,
                "classification": classification,
                "confidence": float(pred_confidence)
            }
            
        except NoFile as e:
            print(f"File not found error: {str(e)}")
            raise HTTPException(status_code=404, detail="File not found")
        except Exception as e:
            print(f"Error during classification process: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in classification endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error during classification: {str(e)}")
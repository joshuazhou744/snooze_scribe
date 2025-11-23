"""Snooze Scribe FastAPI backend.

This service handles Auth0-authenticated audio uploads stored in MongoDB GridFS
and proxies classification requests to the deployed Hugging Face model API.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from functools import lru_cache
from typing import Dict, List

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gridfs.errors import NoFile
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("snooze_scribe_api")


class Settings(BaseModel):
    auth0_audience: str
    auth0_domain: str
    mongodb_url: str
    model_api_url: str
    cors_origins: List[str]

    @classmethod
    def load(cls) -> "Settings":
        data = {
            "auth0_audience": os.getenv("AUTH0_AUDIENCE", ""),
            "auth0_domain": os.getenv("AUTH0_DOMAIN", ""),
            "mongodb_url": os.getenv("MONGODB_URL", ""),
            "model_api_url": os.getenv("MODEL_API_URL", ""),
        }
        missing = [key for key, value in data.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        default_origins = {
            "https://snooze-scribe.vercel.app",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://snooze-scribe-api-production-7c27c458ab69.herokuapp.com",
        }
        extra_origins = {
            origin.strip()
            for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
            if origin.strip()
        }
        data["cors_origins"] = sorted(default_origins | extra_origins)
        return cls(**data)


SETTINGS = Settings.load()
MONGO_CLIENT = AsyncIOMotorClient(SETTINGS.mongodb_url)
AUDIO_DB = MONGO_CLIENT["audio"]


class ClassificationResponse(BaseModel):
    file_id: str
    classification: str
    confidence: float


class AudioFile(BaseModel):
    file_id: str
    filename: str
    audio_url: str
    classification: str = "unclassified"
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _get_jwks() -> Dict[str, str]:
    jwks_url = f"https://{SETTINGS.auth0_domain}/.well-known/jwks.json"
    response = httpx.get(jwks_url, timeout=10)
    response.raise_for_status()
    keys = response.json().get("keys") or []
    if not keys:
        raise RuntimeError("No JWKS keys returned from Auth0")
    return keys[0]


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=403, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid Authorization header") from exc
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Authorization scheme must be Bearer")
    return token


def verify_jwt_token(authorization: str = Header(default=None)) -> str:
    token = _extract_token(authorization)
    try:
        jwks_key = _get_jwks()
        rsa_key = {
            "kty": jwks_key["kty"],
            "kid": jwks_key["kid"],
            "use": jwks_key["use"],
            "n": jwks_key["n"],
            "e": jwks_key["e"],
        }
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=SETTINGS.auth0_audience,
            issuer=f"https://{SETTINGS.auth0_domain}/",
        )
        return payload["sub"]
    except JWTError as exc:  # pragma: no cover - external dependency
        raise HTTPException(status_code=403, detail=f"Token invalid or expired: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Unexpected JWT verification error")
        raise HTTPException(status_code=403, detail="Token verification error") from exc


def _hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def get_user_gridfs(user_id: str) -> AsyncIOMotorGridFSBucket:
    bucket_name = _hash_user_id(user_id)
    return AsyncIOMotorGridFSBucket(AUDIO_DB, bucket_name=bucket_name)


async def _call_model_api(audio_data: bytes) -> Dict[str, float]:
    if not audio_data:
        raise ValueError("Empty audio payload")

    files = {"file": ("audio.mp4", audio_data, "audio/mp4")}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(SETTINGS.model_api_url, files=files)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        LOGGER.error("Model API error %s: %s", exc.response.status_code, exc.response.text[:200])
        raise HTTPException(status_code=502, detail="Model API request failed") from exc

    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
        if isinstance(data, list) and data:
            payload = data[0]

    classification = payload.get("classification")
    confidence = payload.get("confidence", 0.0)

    if classification is None:
        raise HTTPException(status_code=502, detail="Model API returned an invalid response")

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    return {"classification": classification, "confidence": confidence_value}


app = FastAPI(title="Snooze Scribe API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model_api": SETTINGS.model_api_url}


@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_jwt_token),
) -> Dict[str, str]:
    gridfs_bucket = get_user_gridfs(user_id)
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    buffer = io.BytesIO(audio_data)
    file_id = await gridfs_bucket.upload_from_stream(file.filename or "audio.mp4", buffer)
    return {"message": "Audio uploaded successfully", "_id": str(file_id)}


@app.get("/audio-files", response_model=List[AudioFile])
async def list_audio_files(user_id: str = Depends(verify_jwt_token)) -> List[AudioFile]:
    gridfs_bucket = get_user_gridfs(user_id)
    cursor = gridfs_bucket.find()
    files = await cursor.to_list(None)
    results: List[AudioFile] = []
    for file in files:
        results.append(
            AudioFile(
                file_id=str(file["_id"]),
                filename=file["filename"],
                audio_url=f"audio-file/play/{file['_id']}",
                classification=file.get("classification", "unclassified"),
                confidence=float(file.get("confidence", 0.0) or 0.0),
            )
        )
    return results


@app.get("/audio-file/play/{file_id}")
async def play_audio(file_id: str, user_id: str = Depends(verify_jwt_token)) -> StreamingResponse:
    gridfs_bucket = get_user_gridfs(user_id)
    try:
        grid_out = await gridfs_bucket.open_download_stream(ObjectId(file_id))
        return StreamingResponse(grid_out, media_type="audio/mp4")
    except NoFile as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc


@app.delete("/audio-file/{file_id}")
async def delete_audio_file(file_id: str, user_id: str = Depends(verify_jwt_token)) -> Dict[str, str]:
    gridfs_bucket = get_user_gridfs(user_id)
    try:
        await gridfs_bucket.delete(ObjectId(file_id))
        return {"message": "File deleted"}
    except NoFile as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc


@app.delete("/audio-files/all")
async def delete_all_audio_files(user_id: str = Depends(verify_jwt_token)) -> Dict[str, str]:
    gridfs_bucket = get_user_gridfs(user_id)
    cursor = gridfs_bucket.find()
    files = await cursor.to_list(None)

    delete_count = 0
    for file in files:
        try:
            await gridfs_bucket.delete(file["_id"])
            delete_count += 1
        except Exception:  # pragma: no cover - continue deleting others
            LOGGER.exception("Failed to delete file %s", file["_id"])

    return {"message": f"Deleted {delete_count} audio files"}


@app.post("/classify-audio/{file_id}", response_model=ClassificationResponse)
async def classify_audio(file_id: str, user_id: str = Depends(verify_jwt_token)) -> ClassificationResponse:
    gridfs_bucket = get_user_gridfs(user_id)
    try:
        grid_out = await gridfs_bucket.open_download_stream(ObjectId(file_id))
        audio_data = await grid_out.read()
    except NoFile as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc

    result = await _call_model_api(audio_data)
    classification = result["classification"]
    confidence = result["confidence"]

    user_files_collection = AUDIO_DB[f"{_hash_user_id(user_id)}.files"]
    await user_files_collection.update_one(
        {"_id": ObjectId(file_id)},
        {"$set": {"classification": classification, "confidence": confidence}},
    )

    return ClassificationResponse(file_id=file_id, classification=classification, confidence=confidence)


@app.on_event("shutdown")
async def shutdown_event() -> None:  # pragma: no cover - FastAPI hook
    LOGGER.info("Shutting down Snooze Scribe API")
    MONGO_CLIENT.close()

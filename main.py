import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import process_video

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")

app = FastAPI(title="Travel Vlog Guide API")

# Mobile app එකෙන් call කරන්න පුළුවන් වෙන්න CORS allow කරනවා
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production එකේදී mobile app domain එකට restrict කරන්න
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoRequest(BaseModel):
    video_url: str
    user_lat: float | None = None
    user_lng: float | None = None
    travel_mode: str = "driving"


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Travel Vlog Guide API is running"}


@app.post("/api/process-video")
def process_video_endpoint(request: VideoRequest):
    try:
        result = process_video(
            video_url=request.video_url,
            gemini_key=GEMINI_KEY,
            user_lat=request.user_lat,
            user_lng=request.user_lng,
            travel_mode=request.travel_mode,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#/analyze-text = classify one single text segment
#/analyze-conversation = full conversation pipeline
#/analyze = audio pipeline


import os
import shutil
import uuid
from app.pipeline import analyze_audio, classify_text, analyze_conversation_text
from fastapi import FastAPI, UploadFile, File

from app.config import ENABLE_PRIVACY_FILTER
from app.pipeline import analyze_audio, classify_text
from app.privacy import apply_privacy_filter
from app.schemas import (
    AnalyzeResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)


app = FastAPI(
    title="SentimentFlow API",
    description="Audio and text emotion timeline API for detecting Happy, Neutral, and Frustrated states.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "SentimentFlow is running",
        "available_endpoints": [
            "GET /health",
            "GET /test",
            "POST /analyze-text",
            "POST /analyze",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test", response_model=TextAnalyzeResponse)
def test():
    text = "My name is Daniel and I live in Barcelona. It is three days late and nobody told me why!"

    if ENABLE_PRIVACY_FILTER:
        text = apply_privacy_filter(text)

    return classify_text(text)


@app.post("/analyze-single-text", response_model=TextAnalyzeResponse)
def analyze_text(request: TextAnalyzeRequest):
    text = request.text

    if ENABLE_PRIVACY_FILTER:
        text = apply_privacy_filter(text)

    return classify_text(text)

@app.post("/analyze-conversation")
def analyze_conversation(request: TextAnalyzeRequest):
    return analyze_conversation_text(request.text)


@app.post("/analyze-audio", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    call_id = str(uuid.uuid4())

    os.makedirs("/tmp/sentimentflow", exist_ok=True)
    audio_path = f"/tmp/sentimentflow/{call_id}_{file.filename}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_audio(audio_path)

    return {
        "call_id": call_id,
        **result,
    }
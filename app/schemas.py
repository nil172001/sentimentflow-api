from pydantic import BaseModel
from typing import List, Optional


class TextAnalyzeRequest(BaseModel):
    text: str


class TextAnalyzeResponse(BaseModel):
    text: str
    raw_label: str
    confidence: float
    mapped_mood: str


class MoodSegment(BaseModel):
    timestamp: str
    speaker: str
    text: str
    mood: str

    role_confidence: Optional[str] = None
    role_reason: Optional[str] = None
    role_source: Optional[str] = None
    raw_text_available: bool = False
    classifier_label: Optional[str] = None
    confidence: Optional[float] = None
    decision_source: Optional[str] = None
    reason: Optional[str] = None


class LLMUsage(BaseModel):
    role_calls_used: int
    role_call_limit: int
    emotion_calls_used: int
    emotion_call_limit: int


class AnalyzeResponse(BaseModel):
    call_id: str
    summary: str
    final_verdict: str
    readable_conversation: List[str]
    mood_timeline: List[MoodSegment]
    llm_usage: LLMUsage
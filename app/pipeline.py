import re

from faster_whisper import WhisperModel
from transformers import pipeline

from app.config import (
    WHISPER_MODEL_SIZE,
    CLASSIFIER_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    ENABLE_PRIVACY_FILTER,
    ENABLE_LLM_ROLE_CLASSIFICATION,
    CAMPUSAI_API_KEY,
    MAX_LLM_ROLE_CALLS,
    MAX_LLM_EMOTION_CALLS,
    LLM_REFINEMENT_CONFIDENCE_THRESHOLD,
)

from app.privacy import apply_privacy_filter
from app.llm import assign_speaker_role, refine_emotion_with_llm


_emotion_classifier = None
_whisper_model = None


def get_emotion_classifier():
    global _emotion_classifier

    if _emotion_classifier is None:
        print("[MODEL] Loading emotion classifier...")
        _emotion_classifier = pipeline(
            "text-classification",
            model=CLASSIFIER_MODEL,
            top_k=None,
        )
        print("[MODEL] Emotion classifier loaded.")

    return _emotion_classifier


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        print("[MODEL] Loading Whisper model...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            compute_type="int8",
        )
        print("[MODEL] Whisper model loaded.")

    return _whisper_model


def map_emotion(label: str) -> str:
    label = label.lower()

    emotion_mapping = {
        "joy": "Happy",
        "neutral": "Neutral",
        "anger": "Angry",
        "sadness": "Sad",
        "fear": "Fearful",
        "disgust": "Disgusted",
        "surprise": "Surprised",
    }

    return emotion_mapping.get(label, "Neutral")


def classify_text(text: str) -> dict:
    classifier = get_emotion_classifier()
    scores = classifier(text)[0]
    best = max(scores, key=lambda x: x["score"])

    return {
        "text": text,
        "raw_label": best["label"],
        "confidence": round(float(best["score"]), 3),
        "mapped_mood": map_emotion(best["label"]),
    }


def simple_role_heuristic(text: str) -> dict:
    lowered = text.lower()

    agent_phrases = [
        "thank you for calling",
        "customer support",
        "how can i help",
        "how may i help",
        "let me check",
        "i can help",
        "i will check",
        "i understand your concern",
        "i apologize",
        "sorry about that",
        "we can offer",
        "i can offer",
        "i see the issue",
        "full refund",
        "express shipping",
        "is there anything else",
    ]

    customer_phrases = [
        "my order",
        "my package",
        "i am calling because",
        "i'm calling because",
        "i have a problem",
        "i need help",
        "nobody told me",
        "this is frustrating",
        "i am frustrated",
        "i haven't received",
        "i did not receive",
        "i want to complain",
        "that helps a lot",
        "i appreciate your help",
    ]

    if any(phrase in lowered for phrase in agent_phrases):
        return {
            "speaker": "agent",
            "role_confidence": "low",
            "role_reason": "Heuristic fallback matched agent-like wording.",
            "role_source": "heuristic",
        }

    if any(phrase in lowered for phrase in customer_phrases):
        return {
            "speaker": "customer",
            "role_confidence": "low",
            "role_reason": "Heuristic fallback matched customer-like wording.",
            "role_source": "heuristic",
        }

    return {
        "speaker": "unknown",
        "role_confidence": "low",
        "role_reason": "Heuristic fallback could not identify the role.",
        "role_source": "heuristic",
    }


def get_speaker_role(
    text: str,
    previous_context: str,
    llm_role_calls_used: int,
) -> tuple[dict, int]:
    can_use_llm = (
        ENABLE_LLM_ROLE_CLASSIFICATION
        and CAMPUSAI_API_KEY
        and llm_role_calls_used < MAX_LLM_ROLE_CALLS
    )

    if can_use_llm:
        role_result = assign_speaker_role(
            text=text,
            previous_context=previous_context,
        )
        role_result["role_source"] = "llm"
        return role_result, llm_role_calls_used + 1

    role_result = simple_role_heuristic(text)

    if CAMPUSAI_API_KEY and llm_role_calls_used >= MAX_LLM_ROLE_CALLS:
        role_result["role_reason"] += " LLM role call limit reached."

    return role_result, llm_role_calls_used


def refine_emotion_if_needed(
    text: str,
    previous_context: str,
    role_result: dict,
    classifier_result: dict,
    llm_emotion_calls_used: int,
) -> tuple[str, str | None, str, int]:
    confidence = classifier_result["confidence"]

    can_use_llm = (
        CAMPUSAI_API_KEY
        and confidence < LLM_REFINEMENT_CONFIDENCE_THRESHOLD
        and llm_emotion_calls_used < MAX_LLM_EMOTION_CALLS
    )

    if can_use_llm:
        refined = refine_emotion_with_llm(
            text=text,
            speaker=role_result["speaker"],
            previous_context=previous_context,
            classifier_result=classifier_result,
        )

        return (
            refined["mood"],
            refined["reason"],
            "llm_refined_emotion",
            llm_emotion_calls_used + 1,
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return (
            classifier_result["mapped_mood"],
            "Low confidence classifier result; LLM refinement unavailable or limit reached.",
            "classifier_low_confidence_no_llm",
            llm_emotion_calls_used,
        )

    return (
        classifier_result["mapped_mood"],
        None,
        "classifier",
        llm_emotion_calls_used,
    )


def generate_simple_summary(mood_timeline: list) -> dict:
    if not mood_timeline:
        return {
            "summary": "No speech was detected.",
            "final_verdict": "Unknown",
        }

    customer_segments = [
        item for item in mood_timeline
        if item.get("speaker") == "customer"
    ]

    relevant_segments = customer_segments if customer_segments else mood_timeline

    first_mood = relevant_segments[0]["mood"]
    last_mood = relevant_segments[-1]["mood"]

    positive = ["Happy"]
    negative = ["Angry", "Sad", "Fearful", "Disgusted"]

    if last_mood in positive:
        verdict = "Resolved - Positive"
    elif last_mood in negative:
        verdict = "Unresolved - Negative"
    else:
        verdict = "Neutral"

    return {
        "summary": f"The customer journey started as {first_mood} and ended as {last_mood}.",
        "final_verdict": verdict,
    }


def build_readable_conversation(mood_timeline: list) -> list[str]:
    lines = []

    for item in mood_timeline:
        #timestamp = item.get("timestamp", "")
        speaker = item.get("speaker", "unknown").capitalize()
        mood = item.get("mood", "Unknown")
        text = item.get("text", "")

        lines.append(
            f" {speaker} [{mood}]: {text}"
        )

    return lines


def split_conversation_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def analyze_text_segment(
    text: str,
    previous_context: str,
    timestamp: str,
    llm_role_calls_used: int,
    llm_emotion_calls_used: int,
) -> tuple[dict, int, int]:
    role_result, llm_role_calls_used = get_speaker_role(
        text=text,
        previous_context=previous_context,
        llm_role_calls_used=llm_role_calls_used,
    )

    classifier_result = classify_text(text)

    mood, reason, decision_source, llm_emotion_calls_used = refine_emotion_if_needed(
        text=text,
        previous_context=previous_context,
        role_result=role_result,
        classifier_result=classifier_result,
        llm_emotion_calls_used=llm_emotion_calls_used,
    )

    item = {
        "timestamp": timestamp,
        "speaker": role_result["speaker"],
        "text": text,
        "mood": mood,

        "role_confidence": role_result["role_confidence"],
        "role_reason": role_result["role_reason"],
        "role_source": role_result.get("role_source", "unknown"),
        "raw_text_available": False,
        "classifier_label": classifier_result["raw_label"],
        "confidence": classifier_result["confidence"],
        "decision_source": decision_source,
        "reason": reason,
    }

    return item, llm_role_calls_used, llm_emotion_calls_used


def analyze_conversation_text(raw_text: str) -> dict:
    redacted_text = apply_privacy_filter(raw_text) if ENABLE_PRIVACY_FILTER else raw_text
    segments = split_conversation_text(redacted_text)

    mood_timeline = []
    previous_context = ""

    llm_role_calls_used = 0
    llm_emotion_calls_used = 0

    for i, text in enumerate(segments):
        item, llm_role_calls_used, llm_emotion_calls_used = analyze_text_segment(
            text=text,
            previous_context=previous_context,
            timestamp=f"text-segment-{i + 1}",
            llm_role_calls_used=llm_role_calls_used,
            llm_emotion_calls_used=llm_emotion_calls_used,
        )

        mood_timeline.append(item)
        previous_context += " " + text

    summary = generate_simple_summary(mood_timeline)
    readable_conversation = build_readable_conversation(mood_timeline)

    return {
        "summary": summary["summary"],
        "final_verdict": summary["final_verdict"],
        "readable_conversation": readable_conversation,
        "mood_timeline": mood_timeline,
        "llm_usage": {
            "role_calls_used": llm_role_calls_used,
            "role_call_limit": MAX_LLM_ROLE_CALLS,
            "emotion_calls_used": llm_emotion_calls_used,
            "emotion_call_limit": MAX_LLM_EMOTION_CALLS,
        },
    }


def analyze_audio(audio_path: str) -> dict:
    whisper_model = get_whisper_model()
    segments, _ = whisper_model.transcribe(audio_path)

    mood_timeline = []
    previous_context = ""

    llm_role_calls_used = 0
    llm_emotion_calls_used = 0

    for segment in segments:
        raw_text = segment.text.strip()

        if not raw_text:
            continue

        text = apply_privacy_filter(raw_text) if ENABLE_PRIVACY_FILTER else raw_text

        if not text:
            continue

        item, llm_role_calls_used, llm_emotion_calls_used = analyze_text_segment(
            text=text,
            previous_context=previous_context,
            timestamp=f"{segment.start:.1f}-{segment.end:.1f}",
            llm_role_calls_used=llm_role_calls_used,
            llm_emotion_calls_used=llm_emotion_calls_used,
        )

        mood_timeline.append(item)
        previous_context += " " + text

    summary = generate_simple_summary(mood_timeline)
    readable_conversation = build_readable_conversation(mood_timeline)

    return {
        "summary": summary["summary"],
        "final_verdict": summary["final_verdict"],
        "readable_conversation": readable_conversation,
        "mood_timeline": mood_timeline,
        "llm_usage": {
            "role_calls_used": llm_role_calls_used,
            "role_call_limit": MAX_LLM_ROLE_CALLS,
            "emotion_calls_used": llm_emotion_calls_used,
            "emotion_call_limit": MAX_LLM_EMOTION_CALLS,
        },
    }
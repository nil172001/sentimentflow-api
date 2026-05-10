import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# CampusAI Configuration
# =========================

CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

CAMPUSAI_API_URL = os.getenv(
    "CAMPUSAI_API_URL",
    "https://api.campusai.compute.dtu.dk/v1"
).replace('"', "").rstrip("/")

CAMPUSAI_CHAT_MODEL = (
    os.getenv("CAMPUSAI_CHAT_MODEL")
    or os.getenv("CAMPUSAI_MODEL")
    or "Gemma 4"
).replace('"', "")

CAMPUSAI_EMBED_MODEL = os.getenv(
    "CAMPUSAI_EMBED_MODEL",
    "Nomic Embed Text"
).replace('"', "")

# =========================
# Whisper
# =========================

WHISPER_MODEL_SIZE = "tiny"

# =========================
# Emotion Classifier
# =========================

CLASSIFIER_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LOW_CONFIDENCE_THRESHOLD = 0.65

# =========================
# Privacy
# =========================

ENABLE_PRIVACY_FILTER = True
SPACY_PRIVACY_MODEL = "en_core_web_md"

BANNED_WORDS = [
    "passport",
    "credit card",
    "bank account",
    "iban",
    "social security",
    "id number",
    "national id",
    "personal number",
]

# =========================
# LLM Controls
# =========================

ENABLE_LLM_ROLE_CLASSIFICATION = True

MAX_LLM_ROLE_CALLS = 20
MAX_LLM_EMOTION_CALLS = 10

LLM_REFINEMENT_CONFIDENCE_THRESHOLD = 0.65
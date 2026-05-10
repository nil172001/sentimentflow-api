# SentimentFlow API
### Privacy-Aware Customer Call Emotion Analysis

(Inputs to put in analyze conversation in test.txt and .mp3 files for analyze audio)

---

# 🔄 Project Evolution

The original proposal focused on:
- Speech-to-text transcription
- Basic emotion classification (Happy / Neutral / Frustrated)
- Simple JSON summaries

During development, the project evolved into a more advanced hybrid AI system featuring:
- Privacy-preserving NLP
- Speaker-role detection
- Context-aware emotion refinement
- Structured emotional timelines
- Dockerized deployment
- Unit testing

The final system combines:
- `faster-whisper` for transcription
- RoBERTa emotion classification
- CampusAI Gemma 4 for contextual reasoning
- Regex + spaCy for privacy filtering
- FastAPI + Docker for deployment

---

# 📌 Overview

SentimentFlow API analyzes customer support conversations from:
- 🎙️ Audio files (`.mp3`, `.wav`)
- 📝 Raw conversation text

The pipeline:
1. Transcribes conversations
2. Removes sensitive information
3. Identifies customer vs agent speakers
4. Detects emotional changes
5. Produces structured summaries and emotional timelines

---

# 🚀 Main Features

## ✅ Speech-to-Text
Using:
- `faster-whisper`

Audio is converted into timestamped text segments.

---

## ✅ Privacy Protection

Before any LLM analysis, transcripts are sanitized using:
- Regex redaction
- Banned-word filtering
- spaCy Named Entity Recognition (NER)

Sensitive information such as:
- names
- emails
- phone numbers
- locations
- IDs

is removed before analysis.

---

## ✅ Speaker Role Classification

Using:
- CampusAI Gemma 4

The system identifies whether each segment belongs to:
- the customer
- the support agent

---

## ✅ Hybrid Emotion Analysis

### Transformer Classifier
Using:
- `j-hartmann/emotion-english-distilroberta-base`

Provides fast baseline emotion predictions.

### LLM Refinement
Gemma 4 refines low-confidence predictions using:
- conversational context
- speaker role
- customer support semantics

Supported moods:
- Happy
- Neutral
- Angry
- Sad
- Fearful
- Disgusted
- Surprised

---

## ✅ Readable Conversation Output

Example:

```text
0.0-5.4 | Agent [Neutral]: Hello, thank you for calling customer support.
5.4-12.1 | Customer [Angry]: My order is three days late.
````

---

# 🧠 System Architecture

```text
Audio/Text Input
        ↓
Speech-to-Text (faster-whisper)
        ↓
Privacy Filtering (Regex + spaCy)
        ↓
Speaker Role Classification (Gemma 4)
        ↓
Emotion Classification (RoBERTa)
        ↓
LLM Emotion Refinement (Gemma 4)
        ↓
Structured Timeline + Summary
```

---

# 🛡️ Privacy Pipeline

The privacy engine uses three layers:

### 1. Regex Redaction

Removes:

* emails
* dates
* phone numbers
* ID patterns

### 2. Banned Words

Filters sensitive terms such as:

* passport
* credit card
* IBAN

### 3. spaCy NER

Using:

* `en_core_web_md`

Detects:

* person names
* locations
* organizations

Example:

```text
"My name is Alice and I live in Copenhagen."
```

becomes:

```text
"My name is [NAME] and I live in [LOCATION]."
```

---

# ⚡ LLM Usage Optimization

To avoid excessive token usage:

```python
MAX_LLM_ROLE_CALLS = 20
MAX_LLM_EMOTION_CALLS = 10
```

Long calls automatically fall back to:

* heuristic role detection
* classifier-only emotion analysis

---

# 📦 Installation

## Requirements

* Python 3.10+
* Docker Desktop

---

## Local Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

---

## Docker Setup

```bash
docker compose up --build
```

API available at:

```text
http://localhost:8000/docs
```

---

# ⚙️ Configuration

Main settings are located in:

```python
app/config.py
```

Important options:

* `ENABLE_PRIVACY_FILTER`
* `SPACY_PRIVACY_MODEL`
* `WHISPER_MODEL_SIZE`
* `MAX_LLM_ROLE_CALLS`
* `MAX_LLM_EMOTION_CALLS`

---

# 🔑 CampusAI Configuration

Create a `.env` file:

```env
CAMPUSAI_API_KEY=your_key
CAMPUSAI_API_URL="https://api.campusai.compute.dtu.dk/v1"
CAMPUSAI_MODEL="Gemma 4"
CAMPUSAI_EMBED_MODEL="Nomic Embed Text"
```

The system supports both:

* `CAMPUSAI_MODEL`
* `CAMPUSAI_CHAT_MODEL`

---

# 📡 API Endpoints

## POST `/analyze-audio`

Analyze audio files.

## POST `/analyze-conversation`

Analyze raw conversation text.

## POST `/analyze-text`

Analyze a single text segment.

---

# 🧪 Testing

Run tests with:

```bash
docker compose run --rm sentimentflow-api python -m pytest
```

Current tests validate:

* privacy filtering
* emotion mapping
* summary generation

---

# 📚 Technologies Used

| Component              | Technology               |
| ---------------------- | ------------------------ |
| API                    | FastAPI                  |
| Speech-to-Text         | faster-whisper           |
| Emotion Classification | HuggingFace Transformers |
| LLM                    | CampusAI Gemma 4         |
| Privacy NER            | spaCy                    |
| Containerization       | Docker                   |
| Testing                | Pytest                   |

---

# 💡 Future Improvements

Potential extensions:

* Real-time streaming analysis
* Speaker diarization
* Visualization dashboard
* Multi-language support
* Escalation detection

---

# 👨‍💻 Authors

Nil Mataró i Llobet

Developed as part of the DTU AI/LLM course project focused on:

* Natural Language Processing
* Large Language Models
* Privacy-Aware AI Systems

```
```

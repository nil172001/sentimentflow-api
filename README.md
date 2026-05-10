# SentimentFlow API

### Privacy-Aware Customer Call Emotion Analysis

SentimentFlow API is a Python-based web service for analyzing customer support conversations from either audio files or raw conversation text.

The project includes a small set of fictitious customer support examples in `tests/test.txt`, which can be copied into `/analyze-conversation`. Example `.mp3` files can be used with `/analyze` for audio-based testing.

---

# Project Evolution

The original proposal focused on:

- Speech-to-text transcription
- Basic emotion classification into Happy, Neutral, and Frustrated
- Simple JSON summaries
- Optional LLM-based summarization

During development, the project evolved into a more advanced hybrid AI system featuring:

- Privacy-preserving NLP
- Speaker-role detection
- Context-aware emotion refinement
- Structured emotional timelines
- Dockerized deployment
- Unit testing

The final system combines:

- `faster-whisper` for transcription
- RoBERTa-based emotion classification
- CampusAI Gemma 4 for contextual reasoning
- Regex and spaCy for privacy filtering
- FastAPI and Docker for deployment

---

# Overview

SentimentFlow API analyzes customer support conversations from:

- Audio files (`.mp3`, `.wav`)
- Raw conversation text

The pipeline:

1. Transcribes conversations
2. Removes sensitive information
3. Identifies customer vs agent speakers
4. Detects emotional changes
5. Produces structured summaries and emotional timelines

---

# Main Features

## Speech-to-Text

Audio files are transcribed using `faster-whisper`.

The transcription output is split into timestamped text segments, which are then passed through the privacy and analysis pipeline.

## Privacy Protection

Before any LLM analysis, transcripts are sanitized using:

- Regex redaction
- Banned-word filtering
- spaCy Named Entity Recognition

Sensitive information such as names, emails, phone numbers, locations, and IDs is removed before analysis.

Example:

```text
"My name is Alice and I live in Copenhagen."
````

becomes:

```text
"My name is [NAME] and I live in [LOCATION]."
```

## Speaker Role Classification

CampusAI Gemma 4 is used to identify whether each segment belongs to:

* the customer
* the support agent

If the LLM is unavailable or the maximum number of LLM calls is reached, the system falls back to heuristic role detection.

## Hybrid Emotion Analysis

The system uses a transformer classifier as the first prediction step:

* `j-hartmann/emotion-english-distilroberta-base`

CampusAI Gemma 4 is then used to refine low-confidence predictions using:

* conversational context
* speaker role
* customer support semantics

Supported final moods:

* Happy
* Neutral
* Angry
* Sad
* Fearful
* Disgusted
* Surprised

## Readable Conversation Output

In addition to structured JSON, the API returns a readable conversation format:

```text
Agent [Neutral]: Hello, thank you for calling customer support.
Customer [Angry]: My order is three days late.
```

---

# System Architecture

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

# Privacy Pipeline

The privacy engine uses three layers.

## 1. Regex Redaction

Removes structured sensitive patterns such as:

* emails
* dates
* phone numbers
* ID patterns

## 2. Banned Words

Filters sensitive terms such as:

* passport
* credit card
* bank account
* IBAN

These terms are configured in `app/config.py`.

## 3. spaCy Named Entity Recognition

The system uses:

```text
en_core_web_md
```

to detect entities such as:

* person names
* locations
* organizations

All privacy filtering is performed locally before any text is sent to the LLM.

---

# LLM Usage Optimization

To avoid excessive token usage, the system limits the number of LLM calls:

```python
MAX_LLM_ROLE_CALLS = 20
MAX_LLM_EMOTION_CALLS = 10
```

For long conversations, the pipeline automatically falls back to:

* heuristic role detection
* classifier-only emotion analysis

---

# Dataset and Evaluation

The original proposal mentioned public datasets such as the Amazon Customer Service dataset and Hugging Face Emotion dataset for potential fine-tuning or benchmarking.

The current implementation does not perform custom model fine-tuning. Instead, it uses:

* pretrained transformer models
* CampusAI LLM refinement
* synthetic/manual customer support conversations for qualitative testing

A small fictitious dataset is included in:

```text
tests/test.txt
```

These examples cover:

* frustrated customers
* happy resolutions
* privacy redaction cases
* fearful or surprised customers
* longer multi-turn conversations

Evaluation is performed through:

* qualitative testing in the FastAPI Swagger interface
* unit tests for core pipeline logic

The unit tests validate:

* privacy filtering
* emotion mapping
* summary generation

---

# Installation

## Requirements

* Python 3.10+
* Docker Desktop

## Local Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Docker Setup

Build and run the service:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000/docs
```

---

# Configuration

Main settings are located in:

```text
app/config.py
```

Important options:

* `ENABLE_PRIVACY_FILTER`
* `SPACY_PRIVACY_MODEL`
* `WHISPER_MODEL_SIZE`
* `MAX_LLM_ROLE_CALLS`
* `MAX_LLM_EMOTION_CALLS`

## CampusAI Configuration

Create a `.env` file in the project root:

```env
CAMPUSAI_API_KEY=your_key
CAMPUSAI_API_URL="https://api.campusai.compute.dtu.dk/v1"
CAMPUSAI_MODEL="Gemma 4"
CAMPUSAI_EMBED_MODEL="Nomic Embed Text"
```

The real `.env` file should not be committed to Git.

An `.env.example` file should be included instead.

The system supports both:

* `CAMPUSAI_MODEL`
* `CAMPUSAI_CHAT_MODEL`

for compatibility.

---

# API Endpoints

## GET `/health`

Health check endpoint.

## POST `/analyze`

Analyzes an uploaded audio file.

Input:

* `.mp3`
* `.wav`

Output includes:

* transcription
* speaker identification
* emotion timeline
* readable conversation
* final verdict

## POST `/analyze-conversation`

Analyzes raw conversation text.

Example input:

```json
{
  "text": "Hello, thank you for calling customer support. My order is three days late and nobody told me why."
}
```

## POST `/analyze-single-text`

Analyzes a single text segment.

This endpoint is mainly intended for debugging the classifier and privacy filter.

---

# Example Output

```json
{
  "summary": "The customer journey started as Angry and ended as Happy.",
  "final_verdict": "Resolved - Positive",
  "readable_conversation": [
    "Agent [Neutral]: Hello, thank you for calling customer support.",
    "Customer [Angry]: My order is three days late."
  ],
  "mood_timeline": [
    {
      "timestamp": "0.0-5.4",
      "speaker": "agent",
      "text": "Hello, thank you for calling customer support.",
      "mood": "Neutral"
    }
  ]
}
```

---

# Testing

Run tests with:

```bash
docker compose run --rm sentimentflow-api python -m pytest
```

Current tests validate:

* privacy filtering
* emotion mapping
* summary generation

---

# Technologies Used

| Component              | Technology                |
| ---------------------- | ------------------------- |
| API                    | FastAPI                   |
| Speech-to-Text         | faster-whisper            |
| Emotion Classification | Hugging Face Transformers |
| LLM                    | CampusAI Gemma 4          |
| Privacy NER            | spaCy                     |
| Containerization       | Docker                    |
| Testing                | Pytest                    |

---

# Future Improvements

Potential extensions:

* real-time streaming analysis
* speaker diarization
* visualization dashboard
* multi-language support
* escalation detection
* quantitative evaluation on public datasets

---

# Author

Nil Mataró i Llobet

Developed as part of the DTU AI/LLM course project focused on Natural Language Processing

```
```

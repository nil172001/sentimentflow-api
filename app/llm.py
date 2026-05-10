import json
import re
from openai import OpenAI

from app.config import (
    CAMPUSAI_API_KEY,
    CAMPUSAI_API_URL,
    CAMPUSAI_CHAT_MODEL,
)


client = None

if CAMPUSAI_API_KEY:
    client = OpenAI(
        api_key=CAMPUSAI_API_KEY,
        base_url=CAMPUSAI_API_URL,
    )


def extract_json_from_text(content: str) -> dict:
    """
    Robustly extracts JSON from LLM responses.
    Handles:
    - pure JSON
    - ```json ... ```
    - extra text before/after JSON
    """

    content = content.strip()

    print("[LLM RAW RESPONSE]", content)

    # Remove markdown code fences if present
    content = content.replace("```json", "").replace("```", "").strip()

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting first {...} block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No valid JSON found in LLM response: {content[:200]}")


def assign_speaker_role(text: str, previous_context: str = "") -> dict:
    if client is None:
        return {
            "speaker": "unknown",
            "role_confidence": "low",
            "role_reason": "LLM disabled because no CampusAI API key is configured.",
        }

    prompt = f"""
You are classifying one segment from a customer support conversation.

Decide who is speaking.

Allowed speaker labels:
- agent
- customer
- unknown

Rules:
- agent: greets, offers help, apologizes, checks order/status, offers refund/solution
- customer: reports a problem, complains, asks about order/bill/delay, expresses frustration
- unknown: unclear

Previous context:
{previous_context}

Current segment:
{text}

Return ONLY valid JSON. No markdown. No explanation outside JSON.

The JSON must be exactly:
{{
  "speaker": "agent",
  "role_confidence": "high",
  "role_reason": "short reason"
}}
"""

    try:
        response = client.chat.completions.create(
            model=CAMPUSAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON and nothing else.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        parsed = extract_json_from_text(content)

        speaker = parsed.get("speaker", "unknown").lower()
        role_confidence = parsed.get("role_confidence", "low").lower()
        role_reason = parsed.get("role_reason", "")

        if speaker not in ["agent", "customer", "unknown"]:
            speaker = "unknown"

        if role_confidence not in ["high", "medium", "low"]:
            role_confidence = "low"

        return {
            "speaker": speaker,
            "role_confidence": role_confidence,
            "role_reason": role_reason,
        }

    except Exception as error:
        return {
            "speaker": "unknown",
            "role_confidence": "low",
            "role_reason": f"LLM failed: {str(error)}",
        }
    
def refine_emotion_with_llm(
    text: str,
    speaker: str,
    previous_context: str,
    classifier_result: dict,
) -> dict:
    if client is None:
        return {
            "mood": classifier_result["mapped_mood"],
            "reason": "LLM emotion refinement disabled because no CampusAI API key is configured.",
        }

    prompt = f"""
You are analyzing emotion in a customer support conversation.

The text has already been anonymized.

Allowed moods:
- Happy
- Neutral
- Angry
- Sad
- Fearful
- Disgusted
- Surprised

Speaker:
{speaker}

Previous context:
{previous_context}

Current segment:
{text}

Baseline classifier result:
{classifier_result}

Rules:
- Agent greetings or support phrases are usually Neutral, not Happy.
- Customer complaints, delays, unresolved issues, or anger are Frustrated.
- Gratitude after a solution is usually Happy.
- If unclear, choose Neutral.

Return ONLY valid JSON. No markdown. No text outside JSON.

JSON format:
{{
"mood": "Happy",
"reason": "short explanation"
}}
"""

    try:
        response = client.chat.completions.create(
            model=CAMPUSAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON and nothing else.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        parsed = extract_json_from_text(content)

        mood = parsed.get("mood", classifier_result["mapped_mood"])

        allowed_moods = [
                        "Happy",
                        "Neutral",
                        "Angry",
                        "Sad",
                        "Fearful",
                        "Disgusted",
                        "Surprised",]

        if mood not in allowed_moods:
            mood = classifier_result["mapped_mood"]

        return {
            "mood": mood,
            "reason": parsed.get("reason", ""),
        }

    except Exception as error:
        return {
            "mood": classifier_result["mapped_mood"],
            "reason": f"LLM emotion refinement failed: {str(error)}",
        }
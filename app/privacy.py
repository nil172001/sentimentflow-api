import re
import spacy
from app.config import BANNED_WORDS, SPACY_PRIVACY_MODEL


try:
    nlp = spacy.load(SPACY_PRIVACY_MODEL)
except Exception as error:
    print(f"[PRIVACY] spaCy model could not be loaded: {error}")
    nlp = None


def apply_privacy_filter(text: str) -> str:
    if not text:
        return text

    patterns = {
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b": "[EMAIL]",
        r"\b\d{2,4}[-/]\d{1,2}[-/]\d{1,4}\b": "[DATE]",
        r"\b(?:\+?\d{1,3})?[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}\b": "[PHONE]",
    }

    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    # Explicit self-introduction detection
    text = re.sub(
        r"\b(my name is|this is)\s+[A-Z][a-zA-Z\-']+",
        r"\1 [NAME]",
        text,
        flags=re.IGNORECASE,
    )

    if BANNED_WORDS:
        banned_pattern = re.compile(
            r"\b(" + "|".join(re.escape(word) for word in BANNED_WORDS) + r")\b",
            flags=re.IGNORECASE,
        )
        text = banned_pattern.sub("[REDACTED]", text)

    if nlp:
        doc = nlp(text)

        for ent in reversed(doc.ents):
            if ent.label_ == "PERSON":
                text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]

            elif ent.label_ in ["GPE", "LOC"]:
                text = text[:ent.start_char] + "[LOCATION]" + text[ent.end_char:]

    return text
from app.pipeline import generate_simple_summary


def test_empty_timeline():
    result = generate_simple_summary([])
    assert result["final_verdict"] == "Unknown"


def test_positive_resolution():
    timeline = [
        {"mood": "Angry"},
        {"mood": "Happy"},
    ]
    result = generate_simple_summary(timeline)
    assert result["final_verdict"] == "Resolved - Positive"


def test_negative_resolution_angry():
    timeline = [
        {"mood": "Neutral"},
        {"mood": "Angry"},
    ]
    result = generate_simple_summary(timeline)
    assert result["final_verdict"] == "Unresolved - Negative"


def test_negative_resolution_sad():
    timeline = [
        {"mood": "Neutral"},
        {"mood": "Sad"},
    ]
    result = generate_simple_summary(timeline)
    assert result["final_verdict"] == "Unresolved - Negative"


def test_neutral_resolution():
    timeline = [
        {"mood": "Angry"},
        {"mood": "Neutral"},
    ]
    result = generate_simple_summary(timeline)
    assert result["final_verdict"] == "Neutral"
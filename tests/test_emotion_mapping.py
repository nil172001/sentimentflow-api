from app.pipeline import map_emotion


def test_joy_maps_to_happy():
    assert map_emotion("joy") == "Happy"


def test_neutral_maps_to_neutral():
    assert map_emotion("neutral") == "Neutral"


def test_anger_maps_to_angry():
    assert map_emotion("anger") == "Angry"


def test_disgust_maps_to_disgusted():
    assert map_emotion("disgust") == "Disgusted"


def test_fear_maps_to_fearful():
    assert map_emotion("fear") == "Fearful"


def test_sadness_maps_to_sad():
    assert map_emotion("sadness") == "Sad"


def test_surprise_maps_to_surprised():
    assert map_emotion("surprise") == "Surprised"


def test_unknown_maps_to_neutral():
    assert map_emotion("unknown_label") == "Neutral"
from dashboard.app import reasons_to_sentence


def test_reasons_to_sentence_plain_language_for_high_and_low():
    reasons = [
        {"field": "suppression_pct", "z_score": 3.1, "direction": "high"},
        {"field": "reporting_delay_days", "z_score": -2.5, "direction": "low"},
    ]
    sentence = reasons_to_sentence(reasons)
    assert "viral load suppression rate" in sentence
    assert "much higher" in sentence
    assert "reporting delay" in sentence
    assert "much lower" in sentence


def test_reasons_to_sentence_empty_list_has_fallback_message():
    assert "Flagged by the model" in reasons_to_sentence([])

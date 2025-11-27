# -*- coding: utf-8 -*-
"""
Integration tests for the SignSpeak-ML API.

Tests validate that the API server is running and responds correctly
to requests. These tests assume the Docker container is already running
at ML_BASE_URL (default: http://localhost:8000).

Usage:
    pytest test_api_integration.py -v
    pytest test_api_integration.py -v -m integration
    pytest test_api_integration.py -v -m unit
"""

import json
import os
from typing import Any, Dict, List
import urllib.request
import urllib.error

import pytest


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

def get_base_url() -> str:
    """Return base API URL from environment or default."""
    return os.environ.get("ML_BASE_URL", "http://localhost:8000")


def health_url() -> str:
    return f"{get_base_url().rstrip('/')}/health"


def predict_url() -> str:
    return f"{get_base_url().rstrip('/')}/api/predict_landmarks"


# -------------------------------------------------------------
# HTTP Helpers
# -------------------------------------------------------------

def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Send GET request and parse JSON response."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        data = resp.read().decode("utf-8")
        return json.loads(data)


def http_post_json(url: str, payload: Any, timeout: float = 30.0) -> Dict[str, Any]:
    """Send POST request with JSON payload and parse JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


# -------------------------------------------------------------
# Test fixtures / sample data
# -------------------------------------------------------------

def make_sample_framedata(num_frames: int = 10) -> List[Dict[str, Any]]:
    """
    Create sample FrameData-compliant JSON for testing.

    Returns a list of frames with synthetic landmarks (2 hands, 21 points each).
    """
    frames: List[Dict[str, Any]] = []
    for i in range(num_frames):
        # Create landmarks for both hands
        hands_landmarks: List[List[Dict[str, float]]] = []
        hands_handedness: List[List[Dict[str, Any]]] = []

        for hand_id in range(2):  # 0: Right, 1: Left
            lm_list: List[Dict[str, float]] = []
            for pt in range(21):
                lm_list.append({
                    "x": float(0.5 + pt * 0.01),
                    "y": float(0.5 + pt * 0.01),
                    "z": float(0.0),
                    "visibility": 1.0,
                })
            hands_landmarks.append(lm_list)

            handedness_label = "Right" if hand_id == 0 else "Left"
            hands_handedness.append([{
                "score": 1.0,
                "index": hand_id,
                "categoryName": handedness_label,
                "displayName": handedness_label,
            }])

        frames.append({
            "timestamp": float(i * 0.033),
            "sequenceNumber": i,
            "receivedAt": float(i * 0.033),
            "landmarks": hands_landmarks,
            "handedness": hands_handedness,
        })

    return frames


# -------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------

@pytest.mark.integration
def test_health_endpoint():
    """Test that /health endpoint returns status OK."""
    response = http_get_json(health_url())
    assert isinstance(response, dict)
    assert response.get("status") == "ok"


@pytest.mark.integration
def test_predict_landmarks_with_synthetic_data():
    """Test /api/predict_landmarks endpoint with synthetic FrameData."""
    frames = make_sample_framedata(num_frames=60)

    # Validate synthetic data is JSON-serializable
    _ = json.dumps(frames)

    # Send to API
    url = predict_url()
    response = http_post_json(url, frames, timeout=60.0)

    # Validate response structure
    assert isinstance(response, dict)
    assert "error" not in response, f"API error: {response.get('error')}"
    assert "results" in response

    results = response.get("results")
    # Results can be empty, string, list, or dict depending on pipeline output
    assert results is not None


@pytest.mark.integration
@pytest.mark.parametrize("num_frames", [10, 60, 120])
def test_predict_landmarks_various_lengths(num_frames: int):
    """Test API with various sequence lengths."""
    frames = make_sample_framedata(num_frames=num_frames)
    url = predict_url()
    response = http_post_json(url, frames, timeout=60.0)

    assert isinstance(response, dict)
    assert "error" not in response
    assert "results" in response


@pytest.mark.integration
def test_predict_landmarks_empty_list():
    """Test API behavior with empty frame list."""
    frames: List[Dict[str, Any]] = []
    url = predict_url()

    # API should handle empty input gracefully
    try:
        response = http_post_json(url, frames, timeout=10.0)
        # If it succeeds, check it returns valid structure
        assert isinstance(response, dict)
        assert "results" in response or "error" in response
    except urllib.error.HTTPError as e:
        # Some APIs may return 400 for empty input — that's acceptable
        assert e.code in (400, 422)


@pytest.mark.integration
def test_predict_landmarks_minimal_sequence():
    """Test API with minimal 1-frame sequence."""
    frames = make_sample_framedata(num_frames=1)
    url = predict_url()
    response = http_post_json(url, frames, timeout=30.0)

    assert isinstance(response, dict)
    # Single frame is too short for segmentation/classification, so API may return error
    # This is expected behavior - we just verify the API doesn't crash
    assert "results" in response or "error" in response
    if "error" in response:
        # Expected errors for too-short sequences
        error_msg = response["error"].lower()
        assert any(keyword in error_msg for keyword in ["bounds", "empty", "short", "insufficient"])


@pytest.mark.integration
def test_predict_landmarks_short_sequence():
    """Test API with a short but realistic sequence (5 frames)."""
    frames = make_sample_framedata(num_frames=5)
    url = predict_url()
    response = http_post_json(url, frames, timeout=30.0)

    assert isinstance(response, dict)
    # 5 frames should be processable (may not produce results due to lack of motion, but shouldn't error)
    assert "results" in response
    # No assertion on error since short sequences may legitimately not detect words


# -------------------------------------------------------------
# Unit Tests (logic, no API calls)
# -------------------------------------------------------------

@pytest.mark.unit
def test_sample_framedata_structure():
    """Unit test: validate synthetic FrameData structure."""
    frames = make_sample_framedata(num_frames=5)

    assert len(frames) == 5
    for frame in frames:
        assert "timestamp" in frame
        assert "sequenceNumber" in frame
        assert "receivedAt" in frame
        assert "landmarks" in frame
        assert "handedness" in frame

        # landmarks: list of 2 hands, each with 21 points
        assert len(frame["landmarks"]) == 2
        for hand in frame["landmarks"]:
            assert len(hand) == 21
            for pt in hand:
                assert "x" in pt
                assert "y" in pt
                assert "z" in pt
                assert "visibility" in pt

        # handedness: list of 2 entries (one per hand)
        assert len(frame["handedness"]) == 2


@pytest.mark.unit
def test_json_serialization():
    """Unit test: ensure sample data is JSON-serializable."""
    frames = make_sample_framedata(num_frames=10)
    json_str = json.dumps(frames)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Round-trip check
    decoded = json.loads(json_str)
    assert len(decoded) == 10


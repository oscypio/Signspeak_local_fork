#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script: Send entire PKL recording to SignSpeak-ML API as single request.

This script loads a .pkl file containing recorded hand landmarks,
converts it to JSON (FrameData format), and sends the entire sequence
to the prediction API in one request. Results are printed to the console.

Usage:
    python demo_predict_full_pkl.py <path_to_pkl_file>

    Or set environment variables:
        ML_BASE_URL=http://localhost:8000
        PKL_PATH=/path/to/recording.pkl

    python demo_predict_full_pkl.py

Example:
    python demo_predict_full_pkl.py recording.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List
import urllib.request
import urllib.error

try:
    import numpy as np  # type: ignore
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

def get_base_url() -> str:
    """Return API base URL from environment or default."""
    return os.environ.get("ML_BASE_URL", "http://localhost:8000")


def health_url() -> str:
    """Return health check endpoint URL."""
    return f"{get_base_url().rstrip('/')}/health"


def predict_url() -> str:
    """Return the prediction endpoint URL."""
    return f"{get_base_url().rstrip('/')}/api/predict_landmarks"


# -------------------------------------------------------------
# PKL Loading & Conversion
# -------------------------------------------------------------

def load_pkl_as_framedata_json(pkl_path: str) -> List[Dict[str, Any]]:
    """
    Load a .pkl recording and convert it to FrameData-compliant JSON.

    Supported keypoint shapes:
    - (T, 42, 4) — flattened 2 hands × 21 points
    - (T, 2, 21, 4) — structured 2 hands, 21 points per hand

    Returns a list of frame dicts ready for API submission.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keypoints = np.array(data["keypoints"])
    T = keypoints.shape[0]

    # Reshape (T, 42, 4) -> (T, 2, 21, 4) if needed
    if keypoints.ndim == 3 and keypoints.shape[1] == 42:
        keypoints = keypoints.reshape(T, 2, 21, 4)

    # Extract or generate timestamps (~30 FPS)
    timestamps = data.get("timestamps")
    if timestamps is None:
        timestamps = np.arange(T, dtype=float) * 0.033
    else:
        timestamps = np.array(timestamps, dtype=float)

    frames_json: List[Dict[str, Any]] = []

    for i in range(T):
        frame_kp = keypoints[i]  # (2, 21, 4)

        hands_landmarks: List[List[Dict[str, float]]] = []
        hands_handedness: List[List[Dict[str, Any]]] = []

        # 0: Right, 1: Left (convention matching MediaPipe)
        for hand_id, hand_points in enumerate(frame_kp):
            lm_list: List[Dict[str, float]] = []
            for (x, y, z, vis) in hand_points:
                lm_list.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "visibility": float(vis),
                })
            hands_landmarks.append(lm_list)

            handedness_label = "Right" if hand_id == 0 else "Left"
            hands_handedness.append([{
                "score": 1.0,
                "index": int(hand_id),
                "categoryName": handedness_label,
                "displayName": handedness_label,
            }])

        frames_json.append({
            "timestamp": float(timestamps[i]),
            "sequenceNumber": int(i),
            "receivedAt": float(timestamps[i]),
            "landmarks": hands_landmarks,
            "handedness": hands_handedness,
        })

    return frames_json


# -------------------------------------------------------------
# API Communication
# -------------------------------------------------------------

def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """GET request and return parsed JSON response."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


def http_post_json(url: str, payload: Any, timeout: float = 120.0) -> Dict[str, Any]:
    """POST JSON payload and return parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


# -------------------------------------------------------------
# Main Demo Logic
# -------------------------------------------------------------

def run_demo(pkl_path: str):
    """
    Load entire PKL, send to API as one request, and print results.
    """
    print(f"=== SignSpeak-ML API Demo (Full Sequence) ===")
    print(f"PKL file: {pkl_path}")
    print(f"API URL: {predict_url()}")
    print()

    # Check API health first
    print("Checking API health...")
    try:
        health_response = http_get_json(health_url(), timeout=5.0)
        if health_response.get("status") == "ok":
            print("  ✓ API is healthy")
        else:
            print(f"  ⚠ Unexpected health response: {health_response}")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        print("  Continuing anyway...")
    print()

    # Load and convert PKL
    print("Loading PKL file...")
    frames = load_pkl_as_framedata_json(pkl_path)
    print(f"  ✓ Loaded {len(frames)} frames")

    # Calculate total duration
    if len(frames) > 0:
        duration = frames[-1]["timestamp"] - frames[0]["timestamp"]
        print(f"  Duration: {duration:.2f} seconds (~{len(frames)/duration:.1f} FPS)")
    print()

    # Validate JSON serializability
    print("Validating JSON structure...")
    try:
        json_str = json.dumps(frames)
        json_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
        print(f"  ✓ JSON valid (size: {json_size_mb:.2f} MB)")
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
        return
    print()

    # Send to API
    print("Sending entire sequence to API...")
    url = predict_url()

    try:
        start_time = time.time()
        response = http_post_json(url, frames, timeout=120.0)
        elapsed = time.time() - start_time

        print(f"  ✓ Request completed (took {elapsed:.2f}s)")
        print()

        # Check for API errors
        if "error" in response:
            print(f"❌ API Error: {response['error']}")
            return

        # Extract and display results
        print("=== API Response ===")
        results = response.get("results", [])

        if isinstance(results, list):
            print(f"Results type: list (length: {len(results)})")
            print()

            if len(results) == 0:
                print("No results returned (empty list)")
            else:
                print("Results:")
                for idx, result in enumerate(results):
                    print(f"  [{idx}] {result}")

        elif isinstance(results, dict):
            print(f"Results type: dict (keys: {list(results.keys())})")
            print()
            print("Results:")
            for key, value in results.items():
                if isinstance(value, str) and len(value) < 100:
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, dict)) and len(str(value)) < 200:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")

        elif isinstance(results, str):
            print(f"Results type: string (length: {len(results)})")
            print()
            if len(results) < 500:
                print(f"Results: {results}")
            else:
                print(f"Results (first 500 chars):\n{results[:500]}...")

        else:
            print(f"Results type: {type(results).__name__}")
            print(f"Results: {results}")

        print()
        print("=== Full Response JSON ===")
        print(json.dumps(response, indent=2))

    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP Error {e.code}: {e.reason}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"  Error details: {error_body}")
        except:
            pass

    except urllib.error.URLError as e:
        print(f"  ❌ Connection Error: {e.reason}")
        print(f"  Make sure the API server is running at {get_base_url()}")

    except Exception as e:
        print(f"  ❌ Unexpected Error: {type(e).__name__}: {e}")

    print()
    print("=== Demo Complete ===")

    # After demo, request model to reset its internal buffer (endpoint: /api/reset_buffer)
    try:
        reset_url = f"{get_base_url().rstrip('/')}/api/reset_buffer"
        reset_resp = http_post_json(reset_url, {}, timeout=30.0)
        print(f"Buffer reset response: {reset_resp}")
    except Exception as e:
        print(f"Failed to reset buffer: {e}")


# -------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Send entire PKL recording to SignSpeak-ML API as single request.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_predict_full_pkl.py recording.pkl
    
    # With custom API URL
    ML_BASE_URL=http://192.168.1.100:8000 python demo_predict_full_pkl.py recording.pkl
    
    # Using environment variable for PKL path
    PKL_PATH=recording.pkl python demo_predict_full_pkl.py
        """
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=os.environ.get("PKL_PATH", ""),
        help="Path to .pkl file (or set PKL_PATH env variable)"
    )

    args = parser.parse_args()

    if not args.pkl_path:
        print("Error: PKL file not specified.")
        print()
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.pkl_path):
        print(f"Error: PKL file not found: {args.pkl_path}")
        sys.exit(1)

    run_demo(pkl_path=args.pkl_path)


if __name__ == "__main__":
    main()

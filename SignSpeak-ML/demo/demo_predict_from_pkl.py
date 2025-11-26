#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script: Send PKL recording to SignSpeak-ML API in windows.

This script loads a .pkl file containing recorded hand landmarks,
converts it to JSON (FrameData format), slices it into overlapping
windows, and sends each window to the prediction API. Results are
printed to the console.

Usage:
    python demo_predict_from_pkl.py <path_to_pkl_file> [--window-size 120] [--stride 30]

    Or set environment variables:
        ML_BASE_URL=http://localhost:8000
        PKL_PATH=/path/to/recording.pkl

    python demo_predict_from_pkl.py

Example:
    python demo_predict_from_pkl.py recording.pkl --window-size 120 --stride 30
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
# Windowing
# -------------------------------------------------------------

def make_windows(
    frames: List[Dict[str, Any]],
    window_size: int,
    stride: int,
    *,
    drop_incomplete: bool = False
) -> List[List[Dict[str, Any]]]:
    """
    Slice frames into overlapping windows.

    Args:
        frames: List of frame dicts (FrameData).
        window_size: Number of frames per window (e.g., 120).
        stride: Step size between windows (e.g., 30).
        drop_incomplete: If True, skip windows shorter than window_size.

    Returns:
        List of windows (each window is a list of frames).
    """
    n = len(frames)
    windows: List[List[Dict[str, Any]]] = []

    if n == 0:
        return windows

    start = 0
    while start < n:
        end = start + window_size
        window = frames[start:end]
        if len(window) < window_size and drop_incomplete:
            start += stride
            if stride <= 0:
                break
            continue
        windows.append(window)
        start += stride
        if stride <= 0:
            break

    return windows


# -------------------------------------------------------------
# API Communication
# -------------------------------------------------------------

def http_post_json(url: str, payload: Any, timeout: float = 60.0) -> Dict[str, Any]:
    """POST JSON payload and return parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


def send_window_to_api(window: List[Dict[str, Any]], url: str) -> Dict[str, Any]:
    """Send a single window to the API and return the response."""
    return http_post_json(url, window)


# -------------------------------------------------------------
# Main Demo Logic
# -------------------------------------------------------------

def run_demo(pkl_path: str, window_size: int, stride: int, drop_incomplete: bool = False):
    """
    Load PKL, slice into windows, send to API, and print results.
    """
    print(f"=== SignSpeak-ML API Demo ===")
    print(f"PKL file: {pkl_path}")
    print(f"API URL: {predict_url()}")
    print(f"Window size: {window_size}, Stride: {stride}, Drop incomplete: {drop_incomplete}")
    print()

    # Load and convert PKL
    print("Loading PKL file...")
    frames = load_pkl_as_framedata_json(pkl_path)
    print(f"Loaded {len(frames)} frames.")

    # Create windows
    print(f"Creating windows (size={window_size}, stride={stride})...")
    windows = make_windows(frames, window_size, stride, drop_incomplete=drop_incomplete)
    print(f"Created {len(windows)} windows.")
    print()

    if len(windows) == 0:
        print("No windows created. Exiting.")
        return

    # --- now collect sentences and word counts across all responses ---
    collected_sentences = []  # list of tuples: (window_idx, sentence_text, word_count)

    # Send each window to API
    url = predict_url()
    for idx, window in enumerate(windows):
        start_frame = window[0]["sequenceNumber"]
        end_frame = window[-1]["sequenceNumber"]
        print(f"--- Window {idx} (frames {start_frame}-{end_frame}, count={len(window)}) ---")

        try:
            start_time = time.time()
            response = send_window_to_api(window, url)
            elapsed = time.time() - start_time

            # Check for API errors
            if "error" in response:
                print(f"  ❌ API Error: {response['error']}")
                continue

            # Extract results
            results = response.get("results", [])
            print(f"  ✓ Success (took {elapsed:.2f}s)")

            # --- existing printing of results preview ---
            if isinstance(results, list):
                if len(results) == 0:
                    print(f"  Results: (empty list)")
                elif len(results) <= 5:
                    print(f"  Results: {results}")
                else:
                    print(f"  Results (first 5): {results[:5]}")
                    print(f"  ... (total {len(results)} items)")
            elif isinstance(results, dict):
                print(f"  Results (dict keys): {list(results.keys())}")
                if len(str(results)) < 200:
                    print(f"  Results: {results}")
            elif isinstance(results, str):
                if len(results) < 200:
                    print(f"  Results: {results}")
                else:
                    print(f"  Results (preview): {results[:200]}...")
            else:
                print(f"  Results type: {type(results)}, value: {results}")

            # --- extract possible sentence(s) from response ---
            def _collect_texts_from_resp(resp):
                texts = []
                # top-level known keys
                for key in ("sentence", "text", "transcription", "prediction", "predicted_text", "decoded"):
                    v = resp.get(key) if isinstance(resp, dict) else None
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())

                # inspect results payload
                res = resp.get("results") if isinstance(resp, dict) else None
                if isinstance(res, str) and res.strip():
                    texts.append(res.strip())
                elif isinstance(res, list):
                    for item in res:
                        if isinstance(item, str) and item.strip():
                            texts.append(item.strip())
                        elif isinstance(item, dict):
                            for key in ("sentence", "text", "label", "prediction", "transcription"):
                                vv = item.get(key)
                                if isinstance(vv, str) and vv.strip():
                                    texts.append(vv.strip())
                elif isinstance(res, dict):
                    for key in ("sentence", "text", "transcription", "prediction"):
                        vv = res.get(key)
                        if isinstance(vv, str) and vv.strip():
                            texts.append(vv.strip())
                return texts

            found_texts = _collect_texts_from_resp(response)
            # Deduplicate while preserving order
            seen = set()
            unique_texts = []
            for t in found_texts:
                if t not in seen:
                    seen.add(t)
                    unique_texts.append(t)

            for t in unique_texts:
                word_count = len([w for w in t.split() if w])
                collected_sentences.append((idx, t, word_count))
                print(f"  Sentence: \"{t}\"  (words: {word_count})")

        except urllib.error.HTTPError as e:
            print(f"  ❌ HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            print(f"  ❌ Connection Error: {e.reason}")
        except Exception as e:
            print(f"  ❌ Unexpected Error: {e}")

        print()

    # --- After processing all windows, print summary of collected sentences ---
    if collected_sentences:
        print("=== Collected sentences ===")
        total_words = 0
        for i, (win_idx, text, wc) in enumerate(collected_sentences, start=1):
            print(f"{i}. (window {win_idx}) [{wc} words] {text}")
            total_words += wc
        print(f"Total sentences: {len(collected_sentences)}, total words: {total_words}")
    else:
        print("No sentences collected from predictions.")

    print("=== Demo Complete ===")


# -------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Send PKL recording to SignSpeak-ML API in windows."
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=os.environ.get("PKL_PATH", ""),
        help="Path to .pkl file (or set PKL_PATH env variable)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=120,
        help="Number of frames per window (default: 120)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=105,
        help="Stride (step size) between windows (default: 30)"
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Drop windows shorter than window_size"
    )

    args = parser.parse_args()

    if not args.pkl_path or not os.path.isfile(args.pkl_path):
        print("Error: PKL file not found or not specified.")
        print("Usage: python demo_predict_from_pkl.py <path_to_pkl> [options]")
        print("   or: Set PKL_PATH environment variable")
        sys.exit(1)

    run_demo(
        pkl_path=args.pkl_path,
        window_size=args.window_size,
        stride=args.stride,
        drop_incomplete=args.drop_incomplete
    )


if __name__ == "__main__":
    main()

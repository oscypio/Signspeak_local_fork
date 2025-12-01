#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SignSpeak Real-time Detection System
Using SlidingWindowDetector with local pipeline

This script captures webcam input, processes hand landmarks in real-time,
and uses SignSpeak's PipelineManager for word detection and sentence building.

Requirements:
    pip install opencv-python mediapipe torch numpy

Usage:
    python realtime_test.py

Controls:
    - Q: Quit
    - SPACE: Manual sentence end (same as detecting PUSH sign)
    - R: Reset current sentence

Author: SignSpeak Team
"""

import sys
import cv2
import numpy as np
import time
import re
import json
import urllib.request
import urllib.error
from collections import deque
from typing import List, Dict, Any, Optional


import mediapipe as mp



# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration for realtime detection system"""

    # --- API Settings ---
    API_BASE_URL = "http://localhost:8000"  # ML API base URL
    API_TIMEOUT = 30.0  # Request timeout in seconds

    # --- Camera Settings ---
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # --- MediaPipe Settings ---
    MP_MIN_DETECTION_CONFIDENCE = 0.65
    MP_MIN_TRACKING_CONFIDENCE = 0.45
    MP_MAX_NUM_HANDS = 2

    # --- Frame Processing ---
    # Optimized for SlidingWindow (stride=10, so send batches of 10 frames)
    BATCH_SIZE = 10  # Send frames to pipeline every N frames
    FRAME_SKIP = 0   # Process every N-th frame (0 = process all)

    # --- Display Settings ---
    SHOW_LANDMARKS = True
    SHOW_DEBUG_INFO = False
    SHOW_RAW_PREDICTIONS = True  # Show scanning/stability info
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2

    # --- Colors (BGR) ---
    COLOR_BG = (40, 40, 40)
    COLOR_TEXT_PRIMARY = (0, 255, 255)  # Cyan
    COLOR_TEXT_SECONDARY = (255, 255, 255)  # White
    COLOR_TEXT_SUCCESS = (0, 255, 0)  # Green
    COLOR_TEXT_WARNING = (0, 165, 255)  # Orange
    COLOR_RAW_PREDICTION = (0, 255, 255)  # Yellow
    COLOR_STABILITY = (0, 165, 255)  # Orange
    COLOR_PROGRESS_BAR = (0, 255, 0)  # Green

    # --- Special Signs ---
    SPECIAL_LABEL = 'PUSH'  # Default special label for ending sentence


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_word_display(word: str) -> str:
    """Remove numbers from word labels (e.g., HOW1 -> HOW)"""
    if not word:
        return ""
    return re.sub(r'\d+', '', word)


def draw_styled_text(img: np.ndarray, text: str, pos: tuple,
                     color: tuple = (255, 255, 255),
                     scale: float = 0.8, thickness: int = 2):
    """Draw text with shadow effect for better visibility"""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS.ms"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    return f"{mins:02d}:{secs:02d}.{ms:02d}"


def http_post_json(url: str, payload: Any, timeout: float = 30.0) -> Dict[str, Any]:
    """Send JSON POST request and return response"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def mediapipe_to_frame_dict(
    mp_results,
    frame_idx: int,
    timestamp: float
) -> Optional[Dict[str, Any]]:
    """
    Convert MediaPipe results to FrameData dict (JSON-serializable)

    Returns None if no hands detected. This avoids sending empty frames to the
    ML API — frames will be added to the buffer only when landmarks are present.

    Args:
        mp_results: MediaPipe Hands processing results
        frame_idx: Sequential frame number
        timestamp: Timestamp in seconds

    Returns:
        FrameData dict when hands detected, otherwise None.
    """
    # If no hands detected, return None (do not send empty frames)
    if not mp_results or not getattr(mp_results, 'multi_hand_landmarks', None):
        return None

    landmarks_list = []
    handedness_list = []

    # Extract hands if detected
    for hand_idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
        if hand_idx >= 2:  # Max 2 hands
            break

        # Extract landmarks
        lm_list = []
        for landmark in hand_landmarks.landmark:
            lm_list.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": getattr(landmark, 'visibility', 1.0)
            })
        landmarks_list.append(lm_list)

        # Extract handedness (safe-guard if multi_handedness smaller)
        if getattr(mp_results, 'multi_handedness', None) and len(mp_results.multi_handedness) > hand_idx:
            handedness = mp_results.multi_handedness[hand_idx]
            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score

            handedness_list.append([{
                "score": hand_score,
                "index": hand_idx,
                "categoryName": hand_label,
                "displayName": hand_label
            }])

    # Return frame dict (with landmarks present)
    return {
        "timestamp": timestamp,
        "sequenceNumber": frame_idx,
        "receivedAt": timestamp,
        "landmarks": landmarks_list,
        "handedness": handedness_list
    }


# ============================================================
# MAIN REALTIME DETECTION CLASS
# ============================================================

class RealtimeDetector:
    """Main class for real-time sign language detection"""

    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = None
        self.cap = None

        # API endpoints
        self.predict_url = f"{config.API_BASE_URL}/api/predict_landmarks"
        self.force_end_url = f"{config.API_BASE_URL}/api/force_end_sentence"
        self.reset_url = f"{config.API_BASE_URL}/api/reset_buffer"

        # State
        self.frame_buffer: List[Dict[str, Any]] = []
        self.frame_count = 0
        self.start_time = time.time()
        self.current_sentence_words: List[str] = []
        self.final_sentence = "Start signing..."
        self.last_detection_time = 0.0

        # Raw predictions tracking (for display)
        self.last_raw_word = None
        self.last_raw_confidence = 0.0
        self.last_stability_count = 0
        self.detector_buffer_size = 0

        # Statistics
        self.total_words_detected = 0
        self.total_sentences = 0
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

    def initialize(self):
        """Initialize all components"""
        print("\n" + "="*60)
        print("SignSpeak Real-time Detection System (API Mode)")
        print("="*60)

        # 1. Check API Connection
        print("\n[1/3] Checking ML API connection...")
        try:
            # Try to ping the API
            test_url = f"{self.config.API_BASE_URL}/docs"
            print(f"  Connecting to: {self.config.API_BASE_URL}")

            # Simple connectivity check
            try:
                urllib.request.urlopen(test_url, timeout=5)
                print(f"  ✓ API server is running")
            except:
                print(f"  ⚠ Warning: Could not reach {test_url}")
                print(f"  Make sure ML API is running:")
                print(f"    Local: uvicorn app.main:app --reload")
                print(f"    Docker: docker-compose up")
                response = input("  Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False

            print(f"  ✓ Predict endpoint: {self.predict_url}")
            print(f"  ✓ Force end endpoint: {self.force_end_url}")
            print(f"  ✓ Reset endpoint: {self.reset_url}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            return False

        # 2. Initialize MediaPipe
        print("\n[2/3] Initializing MediaPipe Hands...")
        try:
            mp_module = mp.solutions.hands
            self.mp_hands = mp_module.Hands(
                static_image_mode=False,
                max_num_hands=self.config.MP_MAX_NUM_HANDS,
                min_detection_confidence=self.config.MP_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.config.MP_MIN_TRACKING_CONFIDENCE
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print(f"  ✓ MediaPipe ready (max {self.config.MP_MAX_NUM_HANDS} hands, confidence {self.config.MP_MIN_DETECTION_CONFIDENCE})")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            return False

        # 3. Open Camera
        print("\n[3/3] Opening camera...")
        try:
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            if not self.cap.isOpened():
                # Try alternative camera
                print(f"  ! Camera {self.config.CAMERA_INDEX} failed, trying camera 1...")
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    print("  ✗ ERROR: No camera found")
                    return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)

            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"  ✓ Camera opened ({actual_width}x{actual_height} @ {actual_fps}fps)")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            return False

        print("\n" + "="*60)
        print("✓ System Ready!")
        print("\nControls:")
        print("  Q         - Quit")
        print("  SPACE     - End sentence manually")
        print("  R         - Reset current sentence")
        print(f"  {self.config.SPECIAL_LABEL} sign - End sentence (automatic)")
        print("="*60 + "\n")

        return True

    def process_frame_batch(self):
        """Send buffered frames to API for processing"""
        if not self.frame_buffer:
            return []

        try:
            # Send to API
            response = http_post_json(
                self.predict_url,
                self.frame_buffer,
                timeout=self.config.API_TIMEOUT
            )

            # Clear buffer after processing
            self.frame_buffer.clear()

            # Extract results
            if "error" in response:
                print(f"API Error: {response['error']}")
                return []

            return response.get("results", [])

        except urllib.error.URLError as e:
            print(f"ERROR: Could not connect to API: {e}")
            print(f"  Make sure ML API is running on {self.config.API_BASE_URL}")
            self.frame_buffer.clear()
            return []
        except Exception as e:
            print(f"ERROR processing batch: {e}")
            import traceback
            traceback.print_exc()
            self.frame_buffer.clear()
            return []

    def update_raw_predictions_state(self):
        """Extract raw prediction state - not available through API (would need separate endpoint)"""
        # NOTE: Raw predictions are internal to the detector and not exposed via API
        # To show this info, we'd need to add a /ml/get_detector_state endpoint
        # For now, just keep last detected word info
        pass

    def handle_detection_results(self, results: List[Dict[str, Any]]):
        """Process detection results from pipeline"""
        # Update raw predictions state first
        self.update_raw_predictions_state()

        for result in results:
            status = result.get('status')

            if status == 'word_added':
                # New word detected
                word = result.get('prediction', '')
                if word and word != self.config.SPECIAL_LABEL:
                    clean_word = clean_word_display(word)
                    self.current_sentence_words.append(clean_word)
                    self.total_words_detected += 1
                    self.last_detection_time = time.time()

                    # Get confidence if available
                    confidence = result.get('confidence', 0.0)
                    print(f"  ➤ Word: {clean_word} (conf: {confidence:.2%})")

            elif status == 'end_of_sentence':
                # Sentence completed
                sentence = result.get('sentence', '')
                if sentence:
                    self.final_sentence = sentence
                    self.total_sentences += 1
                    print(f"\n  ✓ Sentence: {sentence}\n")
                    self.current_sentence_words.clear()

            elif status == 'no_word':
                # No detection this batch (normal)
                pass

    def draw_ui(self, frame: np.ndarray):
        """Draw user interface overlays"""
        height, width = frame.shape[:2]

        # --- Top bar (dark background) ---
        cv2.rectangle(frame, (0, 0), (width, 110), self.config.COLOR_BG, -1)

        # Current words being built
        if self.current_sentence_words:
            words_text = " > ".join(self.current_sentence_words)
        else:
            words_text = "(no words yet)"
        draw_styled_text(
            frame, f"Building: {words_text}",
            (10, 30), self.config.COLOR_TEXT_PRIMARY, 0.6
        )

        # Final sentence (after polishing)
        draw_styled_text(
            frame, f"Sentence: {self.final_sentence}",
            (10, 65), self.config.COLOR_TEXT_SUCCESS, 0.7
        )

        # Statistics bar
        elapsed = time.time() - self.start_time
        stats_text = f"Words: {self.total_words_detected} | Sentences: {self.total_sentences} | Time: {format_timestamp(elapsed)}"
        draw_styled_text(
            frame, stats_text,
            (10, 95), self.config.COLOR_TEXT_SECONDARY, 0.5
        )

        # --- Bottom info panel ---
        if self.config.SHOW_DEBUG_INFO or self.config.SHOW_RAW_PREDICTIONS:
            # Dark panel background
            panel_height = 120
            cv2.rectangle(frame, (0, height - panel_height),
                         (width, height), (30, 30, 30), -1)

            y_offset = height - panel_height + 25

            # Raw prediction (scanning)
            if self.config.SHOW_RAW_PREDICTIONS and self.last_raw_word:
                raw_text = f"► Scanning: {clean_word_display(self.last_raw_word)}"
                conf_text = f" ({self.last_raw_confidence:.0%} conf)"
                draw_styled_text(frame, raw_text + conf_text,
                               (10, y_offset), self.config.COLOR_RAW_PREDICTION, 0.6)
                y_offset += 25

                # Stability progress
                stability_req = 3  # Default from SLIDING_WINDOW_STABILITY_COUNT
                stability_text = f"► Stability: {self.last_stability_count}/{stability_req} windows"

                # Color based on stability
                if self.last_stability_count >= stability_req:
                    stability_color = self.config.COLOR_TEXT_SUCCESS
                else:
                    stability_color = self.config.COLOR_STABILITY

                draw_styled_text(frame, stability_text,
                               (10, y_offset), stability_color, 0.5)
                y_offset += 25
            else:
                y_offset += 50  # Skip if no raw predictions

            # Detector buffer
            if self.config.SHOW_DEBUG_INFO:
                buffer_text = f"► Detector buffer: {self.detector_buffer_size} frames"
                draw_styled_text(frame, buffer_text,
                               (10, y_offset), self.config.COLOR_TEXT_SECONDARY, 0.5)
                y_offset += 25

                # FPS
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                fps_text = f"FPS: {avg_fps:.1f}"
                draw_styled_text(frame, fps_text,
                               (10, y_offset), self.config.COLOR_TEXT_SECONDARY, 0.5)

            # Batch progress bar (right side)
            progress = len(self.frame_buffer) / self.config.BATCH_SIZE
            bar_width = int(200 * progress)
            bar_y = height - 30
            cv2.rectangle(frame, (width - 220, bar_y),
                         (width - 220 + bar_width, bar_y + 15),
                         self.config.COLOR_PROGRESS_BAR, -1)
            cv2.rectangle(frame, (width - 220, bar_y),
                         (width - 20, bar_y + 15),
                         (100, 100, 100), 2)

            # Batch counter text
            batch_text = f"Batch: {len(self.frame_buffer)}/{self.config.BATCH_SIZE}"
            draw_styled_text(frame, batch_text,
                           (width - 210, height - 12),
                           self.config.COLOR_TEXT_SECONDARY, 0.4)

    def manual_end_sentence(self):
        """Manually trigger sentence end (like PUSH sign)"""
        if self.current_sentence_words:
            print("\n  ⚡ Manual sentence end triggered")
            try:
                response = http_post_json(self.force_end_url, {}, timeout=5.0)

                if "error" in response:
                    print(f"  ✗ API Error: {response['error']}")
                    return

                results = response.get('results', [])
                if results:
                    result = results[0]
                    sentence = result.get('sentence', '')
                    if sentence:
                        self.final_sentence = sentence
                        self.total_sentences += 1
                        print(f"  ✓ Sentence: {sentence}\n")
                self.current_sentence_words.clear()
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
        else:
            print("\n  ! No words to end (buffer empty)\n")

    def reset_sentence(self):
        """Reset current sentence without finalizing"""
        print("\n  ⟲ Resetting current sentence")
        self.current_sentence_words.clear()
        self.final_sentence = "Reset. Start signing..."
        try:
            response = http_post_json(self.reset_url, {}, timeout=5.0)

            if "error" in response:
                print(f"  ✗ API Error: {response['error']}\n")
            else:
                print("  ✓ Pipeline buffer cleared\n")
        except Exception as e:
            print(f"  ✗ ERROR resetting pipeline: {e}\n")

    def run(self):
        """Main detection loop"""
        if not self.initialize():
            print("\n✗ Initialization failed. Exiting.")
            return

        # Create window with fixed size (no auto-scaling)
        window_name = 'SignSpeak Realtime'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT)

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("WARNING: Failed to read frame")
                    break

                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)

                # Skip frames if configured
                if self.config.FRAME_SKIP > 0 and self.frame_count % (self.config.FRAME_SKIP + 1) != 0:
                    self.frame_count += 1
                    cv2.imshow('SignSpeak Realtime', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Process with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = self.mp_hands.process(image_rgb)

                # Draw hand landmarks
                if mp_results.multi_hand_landmarks and self.config.SHOW_LANDMARKS:
                    for hand_landmarks in mp_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS
                        )

                # Convert to FrameData dict (only if hands detected)
                frame_data = mediapipe_to_frame_dict(
                    mp_results,
                    self.frame_count,
                    current_time - self.start_time
                )

                # Add to buffer only when landmarks are present
                if frame_data is not None:
                    self.frame_buffer.append(frame_data)

                # Process batch when buffer is full
                if len(self.frame_buffer) >= self.config.BATCH_SIZE:
                    results = self.process_frame_batch()
                    self.handle_detection_results(results)

                # Update raw predictions state (even without new results)
                if len(self.frame_buffer) == 0:  # Just processed
                    self.update_raw_predictions_state()

                # Draw UI
                self.draw_ui(frame)

                # Display
                cv2.imshow('SignSpeak Realtime', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  ⚠ Quit requested by user")
                    break
                elif key == ord(' '):  # Space
                    self.manual_end_sentence()
                elif key == ord('r'):
                    self.reset_sentence()

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user (Ctrl+C)")

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\n" + "="*60)
        print("Shutting down...")
        print("="*60)

        # Process any remaining frames
        if self.frame_buffer:
            print(f"  Processing {len(self.frame_buffer)} remaining frames...")
            try:
                results = self.process_frame_batch()
                self.handle_detection_results(results)
            except Exception as e:
                print(f"  Warning: Error processing final batch: {e}")

        # Print final statistics
        elapsed = time.time() - self.start_time
        print(f"\nSession Statistics:")
        print(f"  Duration: {format_timestamp(elapsed)}")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Words Detected: {self.total_words_detected}")
        print(f"  Sentences: {self.total_sentences}")
        if elapsed > 0:
            print(f"  Avg FPS: {self.frame_count / elapsed:.1f}")

        # Release resources
        if self.cap:
            self.cap.release()
        if self.mp_hands:
            self.mp_hands.close()
        cv2.destroyAllWindows()

        print("\n✓ Cleanup complete. Goodbye!\n")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Main entry point"""
    config = Config()
    detector = RealtimeDetector(config)
    detector.run()


if __name__ == "__main__":
    main()

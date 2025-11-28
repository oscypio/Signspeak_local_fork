"""
Sliding Window Detector for ASL Sign Recognition

Alternative to traditional segmentation that uses overlapping temporal windows
to detect and classify signs without relying on motion-based boundary detection.

Key advantages:
- Independent of segmentation accuracy
- Natural filtering of false positives through temporal stability
- Works directly with the existing classifier
- Can run in parallel or replace traditional segmenter

Author: SignSpeak Team
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

from ..utils.config import settings


class SlidingWindowDetector:
    """
    Sliding window approach for sign word detection.

    Instead of detecting word boundaries through motion analysis, this detector:
    1. Buffers incoming frames
    2. Extracts overlapping fixed-size windows (stride-based)
    3. Classifies each window independently
    4. Tracks prediction stability across consecutive windows
    5. Emits a word when stable predictions are detected

    This approach is more robust to segmentation errors because it doesn't
    rely on precise boundary detection.

    Parameters:
        window_size: Number of frames per classification window (default: 60)
        stride: Frame skip between consecutive windows (default: 10)
        stability_count: Required consecutive windows with same prediction (default: 3)
        min_confidence: Minimum probability to accept prediction (default: 0.5)
        max_buffer_size: Maximum frames to keep in buffer (default: 300)
    """

    def __init__(
        self,
        window_size: int = settings.SLIDING_WINDOW_SIZE,
        stride: int = settings.SLIDING_WINDOW_STRIDE,
        stability_count: int = settings.SLIDING_WINDOW_STABILITY_COUNT,
        min_confidence: float = settings.SLIDING_WINDOW_MIN_CONFIDENCE,
        max_buffer_size: int = settings.SLIDING_WINDOW_MAX_BUFFER,
    ):
        # Configuration
        self.window_size = window_size
        self.stride = stride
        self.stability_count = stability_count
        self.min_confidence = min_confidence
        self.max_buffer_size = max_buffer_size

        # Frame buffer: stores raw frame vectors as they arrive
        self.frame_buffer: deque = deque(maxlen=max_buffer_size)

        # Tracking state
        self.frames_since_last_window = 0  # Count frames since last window processed
        self.last_predicted_word: Optional[str] = None
        self.last_predicted_confidence: float = 0.0
        self.consecutive_count: int = 0  # How many consecutive windows had same prediction

        # Deduplication: track what we've already emitted to avoid duplicates
        # Format: {word: last_frame_index_when_emitted}
        self.emission_history: Dict[str, int] = {}
        self.total_frames_processed: int = 0

        # Cooldown: prevent emitting same word too quickly
        self.cooldown_frames: int = window_size // 2  # Half window as cooldown

    def reset(self):
        """Reset detector state (useful between sentences or sessions)"""
        self.frame_buffer.clear()
        self.frames_since_last_window = 0
        self.last_predicted_word = None
        self.last_predicted_confidence = 0.0
        self.consecutive_count = 0
        self.emission_history.clear()
        self.total_frames_processed = 0

    def add_frame(
        self,
        frame_vec: np.ndarray,
        classifier: Any,
        preparer: Any
    ) -> Optional[Tuple[str, float]]:
        """
        Add a single frame and potentially return a detected word.

        Args:
            frame_vec: Single frame vector (F,) - flattened normalized keypoints
            classifier: ASLClassifier instance with predict_label/predict_proba methods
            preparer: DataPreparer instance with prepare_resampled method

        Returns:
            None if no word detected, or (word, confidence) tuple if word is ready
        """
        # Add frame to buffer
        self.frame_buffer.append(frame_vec)
        self.frames_since_last_window += 1
        self.total_frames_processed += 1

        # Check if we have enough frames for a window
        if len(self.frame_buffer) < self.window_size:
            return None

        # Check if it's time to process a new window (every 'stride' frames)
        if self.frames_since_last_window < self.stride:
            return None

        # Reset stride counter
        self.frames_since_last_window = 0

        # Extract window: take last window_size frames
        window_frames = list(self.frame_buffer)[-self.window_size:]
        window_np = np.array(window_frames, dtype=np.float32)

        # Prepare window for classification (resample to expected length)
        prepared_window = preparer.prepare_resampled(window_np)

        # Classify the window
        proba_dict = classifier.predict_proba(prepared_window)
        predicted_word = max(proba_dict.items(), key=lambda x: x[1])[0]
        confidence = proba_dict[predicted_word]

        # Filter out low-confidence predictions
        if confidence < self.min_confidence:
            # Reset tracking if confidence drops
            self.last_predicted_word = None
            self.consecutive_count = 0
            return None

        # Check stability: is this the same as last prediction?
        if predicted_word == self.last_predicted_word:
            self.consecutive_count += 1
        else:
            # Different prediction - reset counter
            self.consecutive_count = 1
            self.last_predicted_word = predicted_word
            self.last_predicted_confidence = confidence

        # Check if we've reached stability threshold
        if self.consecutive_count >= self.stability_count:
            # Check cooldown: don't emit same word too soon
            if predicted_word in self.emission_history:
                frames_since_emission = self.total_frames_processed - self.emission_history[predicted_word]
                if frames_since_emission < self.cooldown_frames:
                    # Too soon to emit again
                    return None

            # Emit the word!
            self.emission_history[predicted_word] = self.total_frames_processed

            # Keep tracking but allow for next word
            # Don't reset completely - let it transition naturally
            return (predicted_word, confidence)

        return None

    def add_frames_batch(
        self,
        frame_vecs: List[np.ndarray],
        classifier: Any,
        preparer: Any
    ) -> List[Tuple[str, float]]:
        """
        Process multiple frames at once and return all detected words.

        More efficient than calling add_frame repeatedly when you have
        a batch of frames available.

        Args:
            frame_vecs: List of frame vectors
            classifier: ASLClassifier instance
            preparer: DataPreparer instance

        Returns:
            List of (word, confidence) tuples for detected words
        """
        detected_words = []

        for frame_vec in frame_vecs:
            result = self.add_frame(frame_vec, classifier, preparer)
            if result is not None:
                detected_words.append(result)

        return detected_words

    def add_frames_batch_optimized(
        self,
        frame_vecs: List[np.ndarray],
        classifier: Any,
        preparer: Any
    ) -> List[Tuple[str, float]]:
        """
        Optimized batch processing using batch prediction.

        Collects all windows that need classification and processes them
        in a single batch for better performance.

        Args:
            frame_vecs: List of frame vectors
            classifier: ASLClassifier instance
            preparer: DataPreparer instance

        Returns:
            List of (word, confidence) tuples for detected words
        """
        detected_words = []
        windows_to_classify = []
        window_indices = []  # Track which frames triggered which windows

        # First pass: collect frames and identify windows
        for idx, frame_vec in enumerate(frame_vecs):
            self.frame_buffer.append(frame_vec)
            self.frames_since_last_window += 1
            self.total_frames_processed += 1

            # Check if we should process a window
            if len(self.frame_buffer) >= self.window_size and \
               self.frames_since_last_window >= self.stride:

                self.frames_since_last_window = 0

                # Extract and prepare window
                window_frames = list(self.frame_buffer)[-self.window_size:]
                window_np = np.array(window_frames, dtype=np.float32)
                prepared = preparer.prepare_resampled(window_np)

                windows_to_classify.append(prepared)
                window_indices.append(idx)

        # Batch classify all windows
        if windows_to_classify:
            proba_dicts = classifier.predict_proba_batch(windows_to_classify)

            # Process predictions
            for proba_dict in proba_dicts:
                predicted_word = max(proba_dict.items(), key=lambda x: x[1])[0]
                confidence = proba_dict[predicted_word]

                # Apply same logic as add_frame
                if confidence < self.min_confidence:
                    self.last_predicted_word = None
                    self.consecutive_count = 0
                    continue

                if predicted_word == self.last_predicted_word:
                    self.consecutive_count += 1
                else:
                    self.consecutive_count = 1
                    self.last_predicted_word = predicted_word
                    self.last_predicted_confidence = confidence

                if self.consecutive_count >= self.stability_count:
                    # Check cooldown
                    if predicted_word in self.emission_history:
                        frames_since_emission = self.total_frames_processed - self.emission_history[predicted_word]
                        if frames_since_emission < self.cooldown_frames:
                            continue

                    self.emission_history[predicted_word] = self.total_frames_processed
                    detected_words.append((predicted_word, confidence))

        return detected_words

    def get_state_info(self) -> Dict[str, Any]:
        """Return current state for debugging/monitoring"""
        return {
            "buffer_size": len(self.frame_buffer),
            "frames_since_last_window": self.frames_since_last_window,
            "last_predicted_word": self.last_predicted_word,
            "last_confidence": self.last_predicted_confidence,
            "consecutive_count": self.consecutive_count,
            "total_frames_processed": self.total_frames_processed,
            "words_emitted": len(self.emission_history),
        }


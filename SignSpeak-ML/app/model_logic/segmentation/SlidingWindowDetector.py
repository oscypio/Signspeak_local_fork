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
        # Use larger cooldown for batch processing (prevent detecting same word multiple times in one batch)
        self.cooldown_frames: int = max(window_size // 2, 30)  # At least 30 frames or half window

    def reset(self):
        """Reset detector state (useful between sentences or sessions)"""
        self.frame_buffer.clear()
        self.frames_since_last_window = 0
        self.last_predicted_word = None
        self.last_predicted_confidence = 0.0
        self.consecutive_count = 0
        self.emission_history.clear()
        self.total_frames_processed = 0

    def _calibrate_confidence(self, proba_dict: Dict[str, float]) -> Tuple[str, float]:
        """
        Calibrate confidence using margin-based approach.

        Instead of using raw max probability, use the margin between
        top-1 and top-2 predictions multiplied by max probability.
        This better reflects true confidence.

        Args:
            proba_dict: {label: probability} from classifier

        Returns:
            (predicted_word, calibrated_confidence)
        """
        if not proba_dict:
            return (None, 0.0)

        # Sort by probability descending
        sorted_items = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)

        top_word, top_prob = sorted_items[0]

        if len(sorted_items) < 2:
            # Only one class, use raw probability
            return (top_word, top_prob)

        second_prob = sorted_items[1][1]

        # Margin-based confidence: (top - second) * top
        # This penalizes predictions where second-best is close
        margin = top_prob - second_prob
        calibrated_conf = margin * top_prob

        return (top_word, calibrated_conf)

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
        # Check if frame is "empty" (no hands detected - all zeros or very low values)
        frame_magnitude = float(np.linalg.norm(frame_vec))
        is_empty_frame = frame_magnitude < 0.01  # Threshold for "no hand detected"

        # Track consecutive empty frames
        if not hasattr(self, '_consecutive_empty_frames'):
            self._consecutive_empty_frames = 0

        if is_empty_frame:
            self._consecutive_empty_frames += 1
        else:
            self._consecutive_empty_frames = 0

        # If too many empty frames, clear buffer to prevent ghost detections
        # This prevents detecting old gestures when hands are not visible
        if self._consecutive_empty_frames > self.window_size:
            print(f"[SILENCE DETECTED] Clearing buffer after {self._consecutive_empty_frames} empty frames (magnitude: {frame_magnitude:.6f})")
            # Keep only last window_size frames to maintain some context
            recent_frames = list(self.frame_buffer)[-min(self.window_size, len(self.frame_buffer)):] if len(self.frame_buffer) > 0 else []
            self.frame_buffer.clear()
            for frame in recent_frames:
                self.frame_buffer.append(frame)
            # Reset state completely to prevent detecting old gestures
            self.last_predicted_word = None
            self.consecutive_count = 0
            self.last_predicted_confidence = 0.0
            self._consecutive_empty_frames = 0  # Reset counter after clearing
            return None  # Don't process this frame

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

        # CHECK: Skip classification if window has too many empty frames
        # This prevents classifying windows with MIX of old gesture frames + new empty frames
        window_magnitudes = np.linalg.norm(window_np, axis=1)
        non_empty_count = np.sum(window_magnitudes > 0.01)
        empty_ratio = 1.0 - (non_empty_count / len(window_magnitudes))

        # If window is mostly empty (>50% empty frames), skip classification
        if empty_ratio > 0.5:
            print(f"[SKIP WINDOW] Too many empty frames: {empty_ratio:.1%} empty ({non_empty_count}/{len(window_magnitudes)} non-empty)")
            # Reset state to prevent detecting old gestures
            self.last_predicted_word = None
            self.consecutive_count = 0
            return None

        # Prepare window for classification (resample to expected length)
        prepared_window = preparer.prepare_resampled(window_np)

        # Classify the window
        proba_dict = classifier.predict_proba(prepared_window)
        predicted_word, confidence = self._calibrate_confidence(proba_dict)

        # Filter out low-confidence predictions
        if confidence < self.min_confidence:
            # Don't reset last_predicted_word - keep it for flush!
            # Only reset consecutive count
            self.consecutive_count = 0
            # Update confidence if it's the same word (for flush to use updated value)
            if predicted_word == self.last_predicted_word:
                self.last_predicted_confidence = confidence
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

            # IMPORTANT: Reset tracking to avoid emitting same word repeatedly
            # This forces detector to wait for new stable prediction
            self.consecutive_count = 0
            self.last_predicted_word = None
            self.last_predicted_confidence = 0.0

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
                predicted_word, confidence = self._calibrate_confidence(proba_dict)

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

        # NEW: Flush at end of batch if enabled
        if settings.FORCE_FLUSH_ON_BATCH_END:
            flushed_result = self.flush_buffer(classifier, preparer)
            if flushed_result is not None:
                detected_words.append(flushed_result)

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

    def flush_buffer(
        self,
        classifier: Any,
        preparer: Any,
        min_confidence: float = None
    ) -> Optional[Tuple[str, float]]:
        """
        Force emission of last word (for end of batch/session).

        IMPROVED VERSION with 3 fallback strategies:
        1. Return last tracked word (most reliable)
        2. Classify current buffer as final window
        3. Try short buffer classification (last resort)

        Args:
            classifier: ASLClassifier instance
            preparer: DataPreparer instance
            min_confidence: Override minimum confidence (default from config)

        Returns:
            (word, confidence) if there's a valid prediction, None otherwise
        """
        min_confidence = min_confidence or settings.FLUSH_MIN_CONFIDENCE

        # Check if we have enough frames at all
        if len(self.frame_buffer) < settings.MIN_FRAMES_FOR_FLUSH:
            return None

        # Strategy 1: Return last tracked word (most reliable)
        if self.last_predicted_word is not None:
            if self.last_predicted_confidence >= min_confidence:
                # Use relaxed cooldown for flush (it's end of batch anyway)
                relaxed_cooldown = max(1, self.cooldown_frames // 2)

                if self.last_predicted_word in self.emission_history:
                    frames_since = self.total_frames_processed - self.emission_history[self.last_predicted_word]
                    if frames_since >= relaxed_cooldown:
                        # Enough time passed, emit it
                        self.emission_history[self.last_predicted_word] = self.total_frames_processed
                        return (self.last_predicted_word, self.last_predicted_confidence)
                else:
                    # Never emitted before, definitely return it
                    self.emission_history[self.last_predicted_word] = self.total_frames_processed
                    return (self.last_predicted_word, self.last_predicted_confidence)

        # Strategy 2: Classify current buffer as final window
        if len(self.frame_buffer) >= self.window_size:
            window_frames = list(self.frame_buffer)[-self.window_size:]
            window_np = np.array(window_frames, dtype=np.float32)
            prepared = preparer.prepare_resampled(window_np)

            proba_dict = classifier.predict_proba(prepared)
            predicted_word, confidence = self._calibrate_confidence(proba_dict)

            if predicted_word and confidence >= min_confidence:
                # Check if not recently emitted
                if predicted_word in self.emission_history:
                    frames_since = self.total_frames_processed - self.emission_history[predicted_word]
                    relaxed_cooldown = max(1, self.cooldown_frames // 2)

                    if frames_since < relaxed_cooldown:
                        # Too recent, skip
                        return None

                # Emit new prediction
                self.emission_history[predicted_word] = self.total_frames_processed
                return (predicted_word, confidence)

        # Strategy 3: Short buffer classification (last resort)
        if settings.MIN_FRAMES_FOR_FLUSH <= len(self.frame_buffer) < self.window_size:
            buffer_np = np.array(list(self.frame_buffer), dtype=np.float32)
            prepared = preparer.prepare_resampled(buffer_np)

            proba_dict = classifier.predict_proba(prepared)
            predicted_word, confidence = self._calibrate_confidence(proba_dict)

            # Use even lower threshold for short buffers
            very_low_threshold = min_confidence * 0.8

            if predicted_word and confidence >= very_low_threshold:
                if predicted_word not in self.emission_history:
                    self.emission_history[predicted_word] = self.total_frames_processed
                    return (predicted_word, confidence)

        return None

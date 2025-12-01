"""
Voting-Based Sliding Window Detector - Improved Implementation

Inspired by session_state.py voting mechanism.
Key improvements:
- Stride=1 (classify every frame)
- Fixed buffer size (60 frames)
- Voting mechanism (22 predictions, requires 16/22)
- Better stability and accuracy

Author: SignSpeak Team
Date: 2025-01-30
"""

import numpy as np
from typing import Optional, Tuple, List, Any
from collections import deque, Counter

from ..utils.config import settings


class SlidingWindowDetector:
    """
    Voting-based sliding window detector for sign language recognition.

    Improvements over previous version:
    - Classifies EVERY frame (stride=1) instead of every 10th
    - Uses voting mechanism (22 predictions, 16 votes required) for stability
    - Fixed buffer size (60 frames) for consistency
    - Better confidence filtering

    Parameters from config:
        - SLIDING_WINDOW_SIZE: Window size (default: 60)
        - SLIDING_WINDOW_STRIDE: Frames between windows (default: 1)
        - SLIDING_WINDOW_VOTING_SIZE: Voting window size (default: 22)
        - SLIDING_WINDOW_VOTE_THRESHOLD: Required votes (default: 16)
        - MIN_CONFIDENCE_THRESHOLD: Min confidence (default: 0.75)
    """

    def __init__(
        self,
        window_size: int = settings.SLIDING_WINDOW_SIZE,
        stride: int = settings.SLIDING_WINDOW_STRIDE,
        voting_size: int = getattr(settings, 'SLIDING_WINDOW_VOTING_SIZE', 22),
        vote_threshold: int = getattr(settings, 'SLIDING_WINDOW_VOTE_THRESHOLD', 16),
        min_confidence: float = settings.MIN_CONFIDENCE_THRESHOLD,
        max_buffer_size: int = settings.SLIDING_WINDOW_MAX_BUFFER,
    ):
        # Configuration
        self.window_size = window_size
        self.stride = stride
        self.voting_size = voting_size
        self.vote_threshold = vote_threshold
        self.min_confidence = min_confidence
        self.max_buffer_size = max_buffer_size

        # Use voting if enabled
        self.use_voting = getattr(settings, 'SLIDING_WINDOW_USE_VOTING', True)

        # Frame buffer - use deque for efficient FIFO (O(1) instead of O(n))
        self.frame_buffer: deque = deque(maxlen=window_size)

        # Voting deque - stores recent predictions
        self.voting_deque: deque = deque(maxlen=voting_size)

        # Confidence deque - tracks confidence for each prediction in voting
        self.confidence_deque: deque = deque(maxlen=voting_size)

        # Window tracking
        self.frames_since_last_window = 0

        # Emission tracking
        self.last_emitted_word: Optional[str] = None
        self.frames_since_emission: int = 0
        self.total_frames: int = 0

    def reset(self):
        """Reset all state"""
        self.frame_buffer.clear()
        self.voting_deque.clear()
        self.confidence_deque.clear()
        self.frames_since_last_window = 0
        self.last_emitted_word = None
        self.frames_since_emission = 0
        self.total_frames = 0

    def add_frame(
        self,
        frame_vec: np.ndarray,
        classifier: Any,
        preparer: Any
    ) -> Optional[Tuple[str, float]]:
        """
        Add a single frame and potentially return a detected word.

        Args:
            frame_vec: Frame vector (168,) - flattened landmarks
            classifier: Classifier with predict_proba method
            preparer: DataPreparer with prepare_resampled method

        Returns:
            (word, confidence) if word detected, None otherwise
        """
        # Add frame to buffer (deque auto-drops oldest with maxlen - O(1))
        self.frame_buffer.append(frame_vec)

        self.frames_since_last_window += 1
        self.total_frames += 1
        self.frames_since_emission += 1

        # Check if we have enough frames for a window
        if len(self.frame_buffer) < self.window_size:
            return None

        # Check if it's time to process a window
        if self.frames_since_last_window < self.stride:
            return None

        # Reset stride counter
        self.frames_since_last_window = 0

        # Extract window (all frames in buffer)
        window = np.array(self.frame_buffer, dtype=np.float32)

        # CRITICAL: Check if window has enough non-empty frames
        frame_magnitudes = np.linalg.norm(window, axis=1)
        non_empty_frames = np.sum(frame_magnitudes > 0.01)
        empty_ratio = 1.0 - (non_empty_frames / len(frame_magnitudes))

        # Skip if window is mostly empty (>50% empty frames) - less aggressive
        if empty_ratio > 0.5:
            # Add "UNCERTAIN" to voting deque (like session_state)
            if self.use_voting:
                self.voting_deque.append("UNCERTAIN")
                self.confidence_deque.append(0.0)
            return None

        # Prepare window for classification
        prepared = preparer.prepare_resampled(window)

        # Classify
        proba_dict = classifier.predict_proba(prepared)

        # Get top prediction
        if not proba_dict:
            if self.use_voting:
                self.voting_deque.append("UNCERTAIN")
                self.confidence_deque.append(0.0)
            return None

        predicted_word = max(proba_dict, key=proba_dict.get)
        confidence = proba_dict[predicted_word]

        # Add to voting deque (even if low confidence) + track confidence
        if self.use_voting:
            if confidence >= self.min_confidence:
                self.voting_deque.append(predicted_word)
                self.confidence_deque.append(confidence)
            else:
                self.voting_deque.append("UNCERTAIN")
                self.confidence_deque.append(0.0)

        # Check voting consensus
        if self.use_voting and len(self.voting_deque) == self.voting_size:
            vote_counts = Counter(self.voting_deque)
            top_word, count = vote_counts.most_common(1)[0]

            # Check if winner has enough votes
            if top_word != "UNCERTAIN" and count >= self.vote_threshold:
                # Longer cooldown for stride=1 (prevents rapid re-detection)
                cooldown_frames = 45 if self.stride == 1 else 30

                # Check if this is new word (not recently emitted)
                if top_word != self.last_emitted_word or self.frames_since_emission >= cooldown_frames:
                    # Emit word!
                    self.last_emitted_word = top_word
                    self.frames_since_emission = 0

                    # Calculate average confidence from winner's votes
                    winner_confidences = [
                        self.confidence_deque[i]
                        for i, pred in enumerate(self.voting_deque)
                        if pred == top_word
                    ]
                    word_confidence = np.mean(winner_confidences) if winner_confidences else 0.5

                    return (top_word, word_confidence)

        return None

    def add_frames_batch(
        self,
        frame_vecs: List[np.ndarray],
        classifier: Any,
        preparer: Any
    ) -> List[Tuple[str, float]]:
        """
        Process multiple frames and return all detected words.

        Args:
            frame_vecs: List of frame vectors
            classifier: Classifier instance
            preparer: DataPreparer instance

        Returns:
            List of (word, confidence) tuples
        """
        detected = []

        for frame_vec in frame_vecs:
            result = self.add_frame(frame_vec, classifier, preparer)
            if result is not None:
                detected.append(result)

        return detected

    def add_frames_batch_optimized(
        self,
        frame_vecs: List[np.ndarray],
        classifier: Any,
        preparer: Any
    ) -> List[Tuple[str, float]]:
        """
        Optimized batch processing (for compatibility with PipelineManager).
        In this simple implementation, just call add_frames_batch.
        """
        return self.add_frames_batch(frame_vecs, classifier, preparer)

    def flush_buffer(
        self,
        classifier: Any,
        preparer: Any,
        min_confidence: float = None
    ) -> Optional[Tuple[str, float]]:
        """
        Force emission of last word at end of batch.

        Args:
            classifier: Classifier instance
            preparer: DataPreparer instance
            min_confidence: Override min confidence (optional)

        Returns:
            (word, confidence) if there's a word to emit, None otherwise
        """
        # Check voting deque
        if self.use_voting and len(self.voting_deque) >= self.vote_threshold:
            vote_counts = Counter(self.voting_deque)
            top_word, count = vote_counts.most_common(1)[0]

            min_conf = min_confidence or (self.min_confidence * 0.7)

            if top_word != "UNCERTAIN" and count >= (self.vote_threshold * 0.7):
                if top_word != self.last_emitted_word or self.frames_since_emission >= 15:
                    self.last_emitted_word = top_word
                    self.frames_since_emission = 0
                    return (top_word, 0.8)

        return None

    def get_state_info(self):
        """Return current state for debugging"""
        return {
            "buffer_size": len(self.frame_buffer),
            "voting_deque_size": len(self.voting_deque),
            "confidence_deque_size": len(self.confidence_deque),
            "frames_since_last_window": self.frames_since_last_window,
            "last_emitted_word": self.last_emitted_word,
            "total_frames": self.total_frames,
            "use_voting": self.use_voting,
        }


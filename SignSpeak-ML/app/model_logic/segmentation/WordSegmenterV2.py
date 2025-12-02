# -*- coding: utf-8 -*-
"""
***OBSOLETE***

WordSegmenterV2: High-ROI improvements for word segmentation.

Key features:
- Hysteresis thresholds (HIGH/LOW) to avoid flapping at boundaries
- Adaptive LOW threshold via rolling quantile over recent EMA motion
- Tail refinement: cut segment end at last significant motion point

Contract:
- add_frame(frame_vec: np.ndarray) -> None | np.ndarray (T, F)
- Returns a completed segment as np.ndarray when end detected, otherwise None

Notes:
- Designed for flattened normalized keypoints vectors (T, F)
- EMA-based motion estimation; thresholds updated online
- Keep a small silence tail in buffer to preserve context between words
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
from collections import deque

from ..utils.config import settings


class WordSegmenterV2:
    """
    Improved segmenter with:
      - hysteresis thresholds
      - adaptive low threshold (rolling quantile)
      - refined tail trimming at word end

    Parameters:
      - silence_frames: number of consecutive low-motion frames to confirm end
      - min_word_frames: minimum frames required to return a valid segment
      - ema_alpha: EMA smoothing factor for motion
      - hist_len: number of recent EMA samples for adaptive threshold
      - low_pctl: quantile for low threshold (e.g., 0.5 for median)
      - low_floor: minimum allowed LOW threshold
      - high_multiplier: HIGH threshold = max(HIGH floor, LOW * high_multiplier)
      - high_floor: absolute lower bound for HIGH
      - tail_margin: additive margin above LOW to locate last significant motion
    """

    def __init__(
        self,
        *,
        silence_frames: int = settings.SILENCE_FRAMES,
        min_word_frames: int = settings.MIN_WORD_FRAMES,
        ema_alpha: float = settings.EMA_ALPHA,
        hist_len: int = 30,
        low_pctl: float = 0.5,
        low_floor: float = settings.MOTION_THRESHOLD,
        high_multiplier: float = 1.5,
        high_floor: float = None,
        tail_margin: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.silence_frames = int(silence_frames)
        self.min_word_frames = int(min_word_frames)
        self.ema_alpha = float(ema_alpha)
        self.hist_len = int(hist_len)
        self.low_pctl = float(low_pctl)
        self.low_floor = float(low_floor)
        self.high_multiplier = float(high_multiplier)
        self.high_floor = float(high_floor) if high_floor is not None else float(low_floor * high_multiplier)
        self.tail_margin = float(tail_margin)
        self.debug = debug

        # Buffers
        self.buffer: List[np.ndarray] = []          # frames buffer
        self.ema_buffer: List[float] = []           # aligned EMA for each frame in buffer
        self.motion_hist: deque[float] = deque(maxlen=self.hist_len)  # recent EMA for adaptive LOW

        # State
        self.prev_frame: Optional[np.ndarray] = None
        self.ema_motion: float = 0.0
        self.state: str = "SILENT"  # or "MOVING"
        self.silence_count: int = 0

    # -------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------

    def _update_thresholds(self) -> tuple[float, float]:
        """Compute (LOW, HIGH) thresholds based on recent EMA history."""
        if len(self.motion_hist) == 0:
            low = self.low_floor
        else:
            # rolling quantile
            hist = np.fromiter(self.motion_hist, dtype=float)
            q = float(np.quantile(hist, self.low_pctl))
            low = max(q, self.low_floor)

        high = max(self.high_floor, low * self.high_multiplier)
        return low, high

    def _refine_tail_cut_index(self, low_thr: float) -> int:
        """
        Find the last index in ema_buffer where motion is above (low_thr + tail_margin).
        Returns the cut index (inclusive). If none found, returns len(buffer) - silence_frames - 1.
        """
        if not self.ema_buffer:
            return max(0, len(self.buffer) - self.silence_frames - 1)

        target = low_thr + self.tail_margin
        # Exclude the trailing silence frames when searching for last motion
        end_search = max(0, len(self.ema_buffer) - self.silence_frames)
        for i in range(end_search - 1, -1, -1):
            if self.ema_buffer[i] > target:
                return i
        # Fallback: cut just before silence tail
        return max(0, len(self.buffer) - self.silence_frames - 1)

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def add_frame(self, frame_vec: np.ndarray):
        """
        Consume next frame vector (F,) and return a completed segment (T,F) if an end is detected.
        Otherwise return None.
        """
        if self.prev_frame is None:
            self.prev_frame = frame_vec
            self.buffer.append(frame_vec)
            # initial motion is 0 at first frame
            self.ema_motion = self.ema_alpha * 0.0 + (1.0 - self.ema_alpha) * self.ema_motion
            self.ema_buffer.append(self.ema_motion)
            self.motion_hist.append(self.ema_motion)
            return None

        # Movement magnitude
        movement = float(np.linalg.norm(frame_vec - self.prev_frame))

        # EMA smoothing
        self.ema_motion = self.ema_alpha * movement + (1.0 - self.ema_alpha) * self.ema_motion

        # Update state buffers
        self.prev_frame = frame_vec
        self.buffer.append(frame_vec)
        self.ema_buffer.append(self.ema_motion)
        self.motion_hist.append(self.ema_motion)

        # Update thresholds
        low_thr, high_thr = self._update_thresholds()

        # Optional debug
        if self.debug:
            print(f"ema={self.ema_motion:.6f}, low={low_thr:.6f}, high={high_thr:.6f}, state={self.state}, silence={self.silence_count}, buf={len(self.buffer)}")

        # State machine with hysteresis
        if self.state == "SILENT":
            if self.ema_motion >= high_thr:
                # Enter moving state
                self.state = "MOVING"
                self.silence_count = 0
            # else remain silent
            return None

        # MOVING state
        if self.ema_motion <= low_thr:
            self.silence_count += 1
        else:
            self.silence_count = 0

        # Enough silence -> conclude word
        if self.silence_count >= self.silence_frames:
            # Determine cut index with tail refinement
            cut_idx = self._refine_tail_cut_index(low_thr)
            # Extract frames up to cut index (inclusive)
            word_frames = self.buffer[:cut_idx + 1]

            # Clean too-short gestures
            if len(word_frames) < self.min_word_frames:
                # keep only the last silent frames in buffer
                self.buffer = self.buffer[-self.silence_frames:]
                self.ema_buffer = self.ema_buffer[-self.silence_frames:]
                # Reset to SILENT state after confirming end
                self.state = "SILENT"
                self.silence_count = 0
                return None

            # Convert to numpy
            word_np = np.array(word_frames, dtype=np.float32)

            # Reset buffers: keep the trailing silence tail
            self.buffer = self.buffer[-self.silence_frames:]
            self.ema_buffer = self.ema_buffer[-self.silence_frames:]
            # Reset state
            self.state = "SILENT"
            self.silence_count = 0

            return word_np

        return None


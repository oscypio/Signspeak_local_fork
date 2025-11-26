import numpy as np
from ..utils.config import settings

class WordSegmenter:
    """
    Improved word segmentation for normalized ASL keypoints.
    Designed for:
      - (T, F) flattened vectors (e.g. F = 168 or 168+vel/acc)
      - normalized coordinates (wrist-centered, scaled)
      - low-magnitude movement values (0.001–0.02)

    Uses:
      - smoothed motion signal (EMA)
      - dynamic burst detection
      - adaptive silence threshold
      - minimal word window
    """

    def __init__(
        self,
        motion_threshold=settings.MOTION_THRESHOLD,    # movement < this -> "silence"
        silence_frames=settings.SILENCE_FRAMES,           # stable silence -> word ends
        min_word_frames=settings.MIN_WORD_FRAMES,          # minimal frames for valid word
        burst_multiplier=settings.BURST_MULTIPLIER,       # detect sudden movement surge
        ema_alpha=settings.EMA_ALPHA              # smoothing for movement
    ):
        self.motion_threshold = motion_threshold
        self.silence_frames = silence_frames
        self.min_word_frames = min_word_frames
        self.burst_multiplier = burst_multiplier
        self.ema_alpha = ema_alpha

        self.buffer = []
        self.prev_frame = None
        self.prev_movement = 0.0
        self.silence_count = 0
        self.ema_motion = 0.0

    # -------------------------------------------------------

    def add_frame(self, frame_vec: np.ndarray):
        """
        frame_vec: (F,)
        Returns:
            None
            or np.ndarray(T, F) — completed word
        """

        # First frame
        if self.prev_frame is None:
            self.prev_frame = frame_vec
            self.buffer.append(frame_vec)
            return None

        # Movement magnitude
        movement = float(np.linalg.norm(frame_vec - self.prev_frame))

        # Exponential smoothing
        self.ema_motion = (
            self.ema_alpha * movement
            + (1 - self.ema_alpha) * self.ema_motion
        )

        # Update previous frame and buffer
        self.prev_frame = frame_vec
        self.buffer.append(frame_vec)

        #print(
        #    f"movement={movement:.6f}, ema={self.ema_motion:.6f}, silence={self.silence_count}, buf={len(self.buffer)}")

        # ====================================================
        # 1. Burst detection (adaptive start-of-word boost)
        # ====================================================
        if self.prev_movement > 0 and movement > self.prev_movement * self.burst_multiplier:
            self.silence_count = 0
            self.prev_movement = movement
            return None

        # ====================================================
        # 2. Silence vs movement logic
        # ====================================================
        if self.ema_motion < self.motion_threshold:
            self.silence_count += 1
        else:
            self.silence_count = 0

        # Update for next frame
        self.prev_movement = movement

        # ====================================================
        # 3. Enough silence → conclude previous word
        # ====================================================
        if self.silence_count >= self.silence_frames:

            # Remove trailing silence frames
            word_frames = self.buffer[:-self.silence_frames]

            # Clean too-short gestures
            if len(word_frames) < self.min_word_frames:
                # keep only the last silent frames in buffer
                self.buffer = self.buffer[-self.silence_frames:]
                return None

            # Extract word to return
            word_frames_np = np.array(word_frames, dtype=np.float32)

            # Reset buffer (keep silence tail)
            self.buffer = self.buffer[-self.silence_frames:]

            return word_frames_np

        return None

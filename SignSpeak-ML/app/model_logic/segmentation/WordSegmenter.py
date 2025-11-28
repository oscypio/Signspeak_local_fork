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

        # Track total frames for temporal position
        self.total_frames_processed = 0

    # -------------------------------------------------------

    def add_frame(self, frame_vec: np.ndarray):
        """
        frame_vec: (F,)
        Returns:
            None
            or np.ndarray(T, F) — completed word
        """
        # Increment frame counter
        self.total_frames_processed += 1

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

    def flush_buffer(self, min_frames: int = None) -> np.ndarray:
        """
        Force emission of buffered content (for end of batch/session).

        Args:
            min_frames: Minimum frames required to flush (default from config)

        Returns:
            np.ndarray of buffered frames, or None if buffer too small
        """
        min_frames = min_frames or settings.MIN_FRAMES_FOR_FLUSH

        if len(self.buffer) < min_frames:
            return None

        # Emit entire buffer (don't trim silence - it's the end!)
        word_frames_np = np.array(self.buffer, dtype=np.float32)

        # Clear buffer
        self.buffer = []

        return word_frames_np

    def add_frame_with_alternatives(self, frame_vec: np.ndarray):
        """
        Backwards-compatible alternative to `add_frame`.

        Behaves like `add_frame`, but when a word boundary is detected returns
        a list of candidate segments (np.ndarray). The first element is the
        original segment; following elements are variants produced by
        shifting start/end within +/- SEGMENTER_MAX_SHIFT_FRAMES (uniformly
        sampled up to SEGMENTER_ALTERNATIVES_COUNT). Uses settings from
        app.model_logic.utils.config.

        Returns:
            None or List[np.ndarray]
        """
        from ..utils.config import settings

        # Increment frame counter (same as add_frame)
        self.total_frames_processed += 1

        # Reuse existing add_frame logic to update state without returning
        # the segment. We'll replicate detection here to capture indices.
        if self.prev_frame is None:
            # initialize same as add_frame
            self.prev_frame = frame_vec
            self.buffer.append(frame_vec)
            return None

        movement = float(np.linalg.norm(frame_vec - self.prev_frame))
        self.ema_motion = (
            self.ema_alpha * movement + (1 - self.ema_alpha) * self.ema_motion
        )

        self.prev_frame = frame_vec
        self.buffer.append(frame_vec)

        if self.prev_movement > 0 and movement > self.prev_movement * self.burst_multiplier:
            self.silence_count = 0
            self.prev_movement = movement
            return None

        if self.ema_motion < self.motion_threshold:
            self.silence_count += 1
        else:
            self.silence_count = 0

        self.prev_movement = movement

        if self.silence_count >= self.silence_frames:

            word_frames = self.buffer[:-self.silence_frames]

            if len(word_frames) < self.min_word_frames:
                self.buffer = self.buffer[-self.silence_frames:]
                return None

            # convert to numpy for slicing
            word_np = np.array(word_frames, dtype=np.float32)

            # Determine original start/end indices relative to the buffer
            total_buf = len(self.buffer)
            orig_end = len(word_frames) - 1
            orig_start = 0

            max_shift = int(settings.SEGMENTER_MAX_SHIFT_FRAMES)
            n_variants = int(settings.SEGMENTER_ALTERNATIVES_COUNT)

            variants = []
            # Always include original
            variants.append(word_np)

            # Generate shift-based variants: pairs of (start_shift, end_shift)
            # We'll sample shifts evenly in range [-max_shift, max_shift]
            if n_variants > 1 and max_shift > 0:
                # Create candidate shifts (excluding 0,0 which is original)
                shifts = []
                step = max(1, max_shift // max(1, n_variants - 1))
                # generate symmetric shifts for start and end
                for s in range(step, max_shift + 1, step):
                    shifts.append((-s, 0))
                    shifts.append((s, 0))
                    shifts.append((0, -s))
                    shifts.append((0, s))

                # Deduplicate and limit
                seen = set()
                candidates = []
                for sh in shifts:
                    if len(candidates) >= (n_variants - 1):
                        break
                    if sh in seen:
                        continue
                    seen.add(sh)
                    candidates.append(sh)

                # Create variants from candidates
                for (s_shift, e_shift) in candidates:
                    # Compute new start/end in the original buffer coordinates
                    # start index relative to word_frames
                    start_idx = max(0, orig_start + s_shift)
                    end_idx = min(len(word_frames) - 1, orig_end + e_shift)

                    if end_idx - start_idx + 1 < self.min_word_frames:
                        continue

                    var_np = np.array(word_frames[start_idx:end_idx + 1], dtype=np.float32)
                    variants.append(var_np)

            # Reset buffer (keep silence tail)
            self.buffer = self.buffer[-self.silence_frames:]

            return variants

        return None


    def reset(self):
        """
        Reset segmenter state completely.

        Call this between independent processing batches/windows to avoid
        state contamination from previous data.
        """
        self.buffer = []
        self.prev_frame = None
        self.prev_movement = 0.0
        self.silence_count = 0
        self.ema_motion = 0.0
        # Note: total_frames_processed is NOT reset - it's cumulative

import os

from pydantic.v1 import BaseSettings


# config.py




class Settings(BaseSettings):

    # ==== SEGMENTER ====
    MOTION_THRESHOLD: float = 0.12
    SILENCE_FRAMES: int = 15
    MIN_WORD_FRAMES: int = 8
    BURST_MULTIPLIER: float = 2.00
    EMA_ALPHA = 0.40

    # ==== DATA PREPARER ====
    EXPECTED_FRAMES: int = 60
    ADD_FEATURES: bool = False

    # ===== MODEL ======
    MODEL_PATH: str = os.getenv('MODEL_PATH', "/app/app/model_logic/utils/model_configs/conf(large).pt")
    DEVICE: str = "cpu"
    POLISHING_MODEL_PATH: str = os.getenv('POLISHING_MODEL_PATH',
                                          "/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

    USE_T5:bool = os.getenv('USE_T5', False)

    # ===== OTHER ====
    USE_SEGMENTATOR: bool = os.getenv('USE_SEGMENTER', True)
    SPECIAL_LABEL: str = os.getenv('SPECIAL_LABEL', 'PUSH')

    # ===== SEGMENTER ALTERNATIVES / SCORING =====
    # If True, WordSegmenter will return a list of alternative segmentations
    SEGMENTER_RETURN_ALTERNATIVES: bool = os.getenv('USE_SEGMENTER_ALT', True)

    # How many alternative segment candidates to generate (incl. original)
    SEGMENTER_ALTERNATIVES_COUNT: int = int(os.getenv('SEGMENTER_ALTERNATIVES_COUNT', 10))

    # Maximum shift in frames when generating start/end variants
    SEGMENTER_MAX_SHIFT_FRAMES: int = int(os.getenv('SEGMENTER_MAX_SHIFT_FRAMES', 10))

    # Strategy for variant generation: currently only 'shift_trim' supported
    SEGMENTER_VARIANTS_STRATEGY: str = os.getenv('SEGMENTER_VARIANTS_STRATEGY', 'shift_trim')

    # Scoring method to pick best candidate: 'max_prob' or 'neg_entropy'
    SEGMENTER_SCORING_METHOD: str = os.getenv('SEGMENTER_SCORING_METHOD', 'sum_labels_probs')

    # Early stop if candidate confidence >= this value
    SEGMENTER_EARLY_STOP_PROB: float = float(os.getenv('SEGMENTER_EARLY_STOP_PROB', 0.95))

    # Whether to use batch prediction in classifier for performance (optional)
    SEGMENTER_BATCH_PREDICT: bool = os.getenv('SEGMENTER_BATCH_PREDICT', True)

    # ===== SLIDING WINDOW DETECTOR =====
    # Enable sliding window detection instead of traditional segmenter
    USE_SLIDING_WINDOW: bool = os.getenv('USE_SLIDING_WINDOW', False)

    # Window size for sliding window (should match EXPECTED_FRAMES)
    SLIDING_WINDOW_SIZE: int = int(os.getenv('SLIDING_WINDOW_SIZE', 70))

    # Stride between consecutive windows (lower = more overlap, higher accuracy but slower)
    SLIDING_WINDOW_STRIDE: int = int(os.getenv('SLIDING_WINDOW_STRIDE', 10))

    # Number of consecutive windows with same prediction to confirm word
    SLIDING_WINDOW_STABILITY_COUNT: int = int(os.getenv('SLIDING_WINDOW_STABILITY_COUNT', 3))

    # Minimum confidence threshold to accept a prediction
    SLIDING_WINDOW_MIN_CONFIDENCE: float = float(os.getenv('SLIDING_WINDOW_MIN_CONFIDENCE', 0.6))

    # Maximum buffer size (frames to keep in memory)
    SLIDING_WINDOW_MAX_BUFFER: int = int(os.getenv('SLIDING_WINDOW_MAX_BUFFER', 800))

    # Use batch prediction for sliding windows (recommended for performance)
    SLIDING_WINDOW_BATCH_PREDICT: bool = os.getenv('SLIDING_WINDOW_BATCH_PREDICT', True)


settings = Settings()

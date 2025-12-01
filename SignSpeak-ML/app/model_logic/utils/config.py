import os

from pydantic.v1 import BaseSettings


# config.py




class Settings(BaseSettings):

    # ==== SEGMENTER ====
    MOTION_THRESHOLD: float = 0.12
    SILENCE_FRAMES: int = 6
    MIN_WORD_FRAMES: int = 8
    BURST_MULTIPLIER: float = 2.00
    EMA_ALPHA = 0.40
    # Words below this threshold are ignored (no response)
    MIN_CONFIDENCE_THRESHOLD: float = os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6)

    # ==== DATA PREPARER ====
    EXPECTED_FRAMES: int = 60
    ADD_FEATURES: bool = False

    # Normalization method: 'wrist' (default) or 'hybrid' (preserves spatial context)
    USE_HYBRID_NORMALIZATION: bool = os.getenv('USE_HYBRID_NORMALIZATION', True)

    # ===== MODEL ======
    MODEL_PATH: str = os.getenv('MODEL_PATH', "/app/app/model_logic/utils/model_configs/conf(large).pt")
    DEVICE: str = "cpu"
    POLISHING_MODEL_PATH: str = os.getenv('POLISHING_MODEL_PATH',
                                          "/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

    USE_T5:bool = os.getenv('USE_T5', False)

    # ===== OTHER ====
    USE_SEGMENTATOR: bool = os.getenv('USE_SEGMENTER', False)
    SPECIAL_LABEL: str = os.getenv('SPECIAL_LABEL', 'PUSH')

    # ===== SEGMENTER ALTERNATIVES / SCORING =====
    # If True, WordSegmenter will return a list of alternative segmentations
    SEGMENTER_RETURN_ALTERNATIVES: bool = os.getenv('USE_SEGMENTER_ALT', True)

    # How many alternative segment candidates to generate (incl. original)
    SEGMENTER_ALTERNATIVES_COUNT: int = int(os.getenv('SEGMENTER_ALTERNATIVES_COUNT', 30))

    # Maximum shift in frames when generating start/end variants
    SEGMENTER_MAX_SHIFT_FRAMES: int = int(os.getenv('SEGMENTER_MAX_SHIFT_FRAMES', 30))

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
    USE_SLIDING_WINDOW: bool = os.getenv('USE_SLIDING_WINDOW', True)

    # Window size for sliding window (should match EXPECTED_FRAMES)
    SLIDING_WINDOW_SIZE: int = int(os.getenv('SLIDING_WINDOW_SIZE', 60))

    # Stride between consecutive windows (1 = classify every frame like voting system)
    SLIDING_WINDOW_STRIDE: int = int(os.getenv('SLIDING_WINDOW_STRIDE', 1))

    # Number of consecutive windows with same prediction to confirm word
    SLIDING_WINDOW_STABILITY_COUNT: int = int(os.getenv('SLIDING_WINDOW_STABILITY_COUNT', 3))

    # Minimum confidence threshold to accept a prediction
    SLIDING_WINDOW_MIN_CONFIDENCE: float = float(os.getenv('SLIDING_WINDOW_MIN_CONFIDENCE', 0.55))

    # Maximum buffer size - fixed at 60 frames (like voting system)
    SLIDING_WINDOW_MAX_BUFFER: int = int(os.getenv('SLIDING_WINDOW_MAX_BUFFER', 60))

    # Use batch prediction for sliding windows (recommended for performance)
    SLIDING_WINDOW_BATCH_PREDICT: bool = os.getenv('SLIDING_WINDOW_BATCH_PREDICT', True)

    # ===== VOTING MECHANISM =====
    # Use voting mechanism instead of simple consecutive counting
    SLIDING_WINDOW_USE_VOTING: bool = True

    # Voting window size - how many recent predictions to consider
    SLIDING_WINDOW_VOTING_SIZE: int = int(os.getenv('SLIDING_WINDOW_VOTING_SIZE', 22))

    # Vote threshold - minimum votes required (out of VOTING_SIZE)
    SLIDING_WINDOW_VOTE_THRESHOLD: int = int(os.getenv('SLIDING_WINDOW_VOTE_THRESHOLD', 15))

    # ===== HYBRID MODE =====
    # Use both segmenter and sliding window, combine results intelligently
    USE_HYBRID_MODE: bool = os.getenv('USE_HYBRID_MODE', False)

    # Strategy for combining segmenter and sliding window results:
    # - 'max_confidence': Choose detection with highest confidence
    # - 'voting': Use majority voting if both detect same word
    # - 'segmenter_primary': Use sliding window only if segmenter has low confidence
    # - 'sliding_primary': Use segmenter only if sliding window has low confidence
    HYBRID_STRATEGY: str = os.getenv('HYBRID_STRATEGY', 'segmenter_primary')

    # Minimum confidence difference to prefer one method over another
    HYBRID_CONFIDENCE_THRESHOLD: float = float(os.getenv('HYBRID_CONFIDENCE_THRESHOLD', 0.1))

    # If both methods agree on the word, boost confidence
    HYBRID_AGREEMENT_BOOST: float = float(os.getenv('HYBRID_AGREEMENT_BOOST', 0.15))

    # ===== HYBRID TEMPORAL MATCHING =====
    # Minimum temporal overlap (IoU) to consider two detections as same segment
    HYBRID_OVERLAP_THRESHOLD: float = float(os.getenv('HYBRID_OVERLAP_THRESHOLD', 0.5))

    # Strategy when no temporal match found:
    # - 'include_all': Add both segmenter-only and sliding-only detections (default)
    # - 'require_both': Discard detections without a match (high precision)
    # - 'prefer_segmenter': Only add segmenter-only detections
    # - 'prefer_sliding': Only add sliding-only detections
    HYBRID_NO_MATCH_STRATEGY: str = os.getenv('HYBRID_NO_MATCH_STRATEGY', 'prefer_segmenter')

    # Word-level deduplication for same words without IoU match - now obsolete (use only for debug)
    HYBRID_WORD_DEDUP_ENABLED: bool = os.getenv('HYBRID_WORD_DEDUP_ENABLED', False)

    # How to choose when deduplicating same word:
    # - 'max_confidence': Keep detection with highest confidence (default)
    # - 'first': Keep first detection chronologically
    # - 'merge': Average confidences
    HYBRID_DEDUP_STRATEGY: str = os.getenv('HYBRID_DEDUP_STRATEGY', 'max_confidence')

    # ===== BUFFER FLUSH =====
    # Force emission of buffered content when batch ends (prevents losing last word)
    FORCE_FLUSH_ON_BATCH_END: bool = True

    # Minimum frames in buffer to force flush (prevents flushing noise)
    MIN_FRAMES_FOR_FLUSH: int = 20

    # Minimum confidence for forced flush (lower than normal to catch uncertain words)
    FLUSH_MIN_CONFIDENCE: float = 0.4

    # ===== WORD DETECTION THRESHOLD =====
    # Minimum confidence required to emit a detected word (0.0 - 1.0)



settings = Settings()

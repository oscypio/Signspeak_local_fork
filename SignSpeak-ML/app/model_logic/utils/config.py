import os

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    """
    SignSpeak ML Configuration

    Organized by component:
    1. Core Detection (Segmenter & Sliding Window)
    2. Hybrid Mode (Adaptive combination)
    3. Model & Preprocessing
    4. Performance & Buffer Management
    """

    # ========================================================================
    # 1. CORE DETECTION SETTINGS
    # ========================================================================

    # ===== GENERAL DETECTION =====

    # Master confidence threshold - filters ALL detections (segmenter, sliding, hybrid)
    # Lower = more detections (higher recall), Higher = fewer false positives (higher precision)
    MIN_CONFIDENCE_THRESHOLD: float = os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6)  # Suggest: 0.75 for production

    # Special label for end-of-sentence (triggers polishing and buffer flush)
    SPECIAL_LABEL: str = os.getenv('SPECIAL_LABEL', 'PUSH')

    # Which detection method to use (only one should be True, or use USE_HYBRID_MODE)
    USE_SEGMENTATOR: bool = os.getenv('USE_SEGMENTER', False)
    USE_SLIDING_WINDOW: bool = os.getenv('USE_SLIDING_WINDOW', True)
    USE_HYBRID_MODE: bool = os.getenv('USE_HYBRID_MODE', False)  # Combines both methods intelligently

    # ===== SEGMENTER (Motion-Based Detection) =====

    # Minimum motion magnitude to detect gesture start (0.0-1.0)
    MOTION_THRESHOLD: float = 0.2  # Lower = more sensitive to motion

    # Number of consecutive low-motion frames to end a segment
    SILENCE_FRAMES: int = 6  # ~0.2s at 30fps

    # Minimum segment length in frames to be considered a valid word
    MIN_WORD_FRAMES: int = 8  # ~0.27s at 30fps

    # Motion burst detection multiplier (for sudden movements)
    BURST_MULTIPLIER: float = 2.00

    # Exponential moving average alpha for motion smoothing
    EMA_ALPHA: float = 0.43



    # --- Segmenter Alternatives (TTA-like approach) ---
    # Generate multiple segment variants for better accuracy
    SEGMENTER_RETURN_ALTERNATIVES: bool = os.getenv('USE_SEGMENTER_ALT', True)
    SEGMENTER_ALTERNATIVES_COUNT: int = int(os.getenv('SEGMENTER_ALTERNATIVES_COUNT', 8))  # Number of variants to test
    SEGMENTER_MAX_SHIFT_FRAMES: int = int(os.getenv('SEGMENTER_MAX_SHIFT_FRAMES', 30))  # Max boundary adjustment
    SEGMENTER_VARIANTS_STRATEGY: str = os.getenv('SEGMENTER_VARIANTS_STRATEGY', 'shift_trim')
    SEGMENTER_SCORING_METHOD: str = os.getenv('SEGMENTER_SCORING_METHOD', 'sum_labels_probs')
    SEGMENTER_EARLY_STOP_PROB: float = float(os.getenv('SEGMENTER_EARLY_STOP_PROB', 0.95))
    SEGMENTER_BATCH_PREDICT: bool = os.getenv('SEGMENTER_BATCH_PREDICT', True)

    # ===== SLIDING WINDOW DETECTOR (Continuous Classification) =====

    # Window size - MUST match EXPECTED_FRAMES (model input size)
    SLIDING_WINDOW_SIZE: int = int(os.getenv('SLIDING_WINDOW_SIZE', 60))  # DON'T CHANGE without retraining

    # Stride - how often to classify (1 = every frame, 5 = every 5th frame)
    # Lower stride = more predictions = better accuracy but slower
    SLIDING_WINDOW_STRIDE: int = int(os.getenv('SLIDING_WINDOW_STRIDE', 1))  # Suggest: 1 for best accuracy, 5 for speed

    # Minimum confidence for individual predictions (before voting)
    SLIDING_WINDOW_MIN_CONFIDENCE: float = float(os.getenv('SLIDING_WINDOW_MIN_CONFIDENCE', 0.51))

    # Buffer size (should equal WINDOW_SIZE)
    SLIDING_WINDOW_MAX_BUFFER: int = int(os.getenv('SLIDING_WINDOW_MAX_BUFFER', 60))

    # Use batch prediction for performance
    SLIDING_WINDOW_BATCH_PREDICT: bool = os.getenv('SLIDING_WINDOW_BATCH_PREDICT', True)

    # --- Voting Mechanism (Provides stability and accuracy) ---
    # Collects recent predictions and requires majority vote to accept word
    SLIDING_WINDOW_USE_VOTING: bool = True

    # How many recent predictions to store for voting
    SLIDING_WINDOW_VOTING_SIZE: int = int(os.getenv('SLIDING_WINDOW_VOTING_SIZE', 20))  # Suggest: 22 for stride=1

    # Minimum votes required (out of VOTING_SIZE) to accept word
    SLIDING_WINDOW_VOTE_THRESHOLD: int = int(os.getenv('SLIDING_WINDOW_VOTE_THRESHOLD', 13))  # 65% by default, suggest: 16 for 80%

    # DEPRECATED: Old stability counting method (use voting instead)
    SLIDING_WINDOW_STABILITY_COUNT: int = int(os.getenv('SLIDING_WINDOW_STABILITY_COUNT', 3))  # Not used with voting

    # ========================================================================
    # 2. HYBRID MODE SETTINGS (Combines Segmenter + Sliding Window)
    # ========================================================================

    # ===== HYBRID STRATEGY =====

    # Combination strategy:
    # - 'adaptive': IoU-based matching with intelligent conflict resolution (RECOMMENDED)
    # - 'max_confidence': Choose detection with highest confidence
    # - 'voting': Majority voting if both detect same word
    # - 'segmenter_primary': Prefer segmenter, use sliding as backup
    # - 'sliding_primary': Prefer sliding, use segmenter as backup
    HYBRID_STRATEGY: str = os.getenv('HYBRID_STRATEGY', 'adaptive')

    # ===== HYBRID TEMPORAL MATCHING =====

    # Minimum IoU (Intersection over Union) to consider detections as matching (0.0-1.0)
    # Higher = stricter matching (fewer matches), Lower = more lenient (more matches)
    HYBRID_OVERLAP_THRESHOLD: float = float(os.getenv('HYBRID_OVERLAP_THRESHOLD', 0.5))  # 50% overlap, suggest: 0.6 for precision

    # Confidence boost when both methods agree on same word
    HYBRID_AGREEMENT_BOOST: float = float(os.getenv('HYBRID_AGREEMENT_BOOST', 0.15))  # +15%, suggest: 0.10-0.20

    # Minimum confidence difference to consider one method "significantly better" in conflicts
    HYBRID_CONFIDENCE_THRESHOLD: float = float(os.getenv('HYBRID_CONFIDENCE_THRESHOLD', 0.1))  # 10%

    # ===== HYBRID SOLO DETECTION (NEW!) =====

    # Multiplier for MIN_CONFIDENCE_THRESHOLD when adding solo detections (not matched by both detectors)
    # Solo detections need slightly lower threshold to avoid missing gestures
    # Actual threshold = MIN_CONFIDENCE_THRESHOLD * HYBRID_SOLO_DETECTION_MULTIPLIER
    # Example: MIN_THRESHOLD=0.75, MULTIPLIER=0.9 → solo needs 0.675
    HYBRID_SOLO_DETECTION_MULTIPLIER: float = float(os.getenv('HYBRID_SOLO_DETECTION_MULTIPLIER', 0.85))  # 90%, suggest: 0.85-0.95

    # ===== HYBRID LEGACY SETTINGS =====
    # These are mostly deprecated but kept for compatibility

    # DEPRECATED: Strategy when no match found (adaptive strategy handles this automatically)
    HYBRID_NO_MATCH_STRATEGY: str = os.getenv('HYBRID_NO_MATCH_STRATEGY', 'prefer_segmenter')

    # DEPRECATED: Word-level deduplication without temporal info (use temporal deduplication in adaptive instead)
    HYBRID_WORD_DEDUP_ENABLED: bool = os.getenv('HYBRID_WORD_DEDUP_ENABLED', False)
    HYBRID_DEDUP_STRATEGY: str = os.getenv('HYBRID_DEDUP_STRATEGY', 'max_confidence')

    # ========================================================================
    # 3. MODEL & PREPROCESSING
    # ========================================================================

    # ===== MODEL PATHS =====

    # Main classification model (GRU-based gesture classifier)
    MODEL_PATH: str = os.getenv('MODEL_PATH', "/app/app/model_logic/utils/model_configs/conf(large).pt")

    # Sentence polishing model (LLM for grammar correction)
    POLISHING_MODEL_PATH: str = os.getenv('POLISHING_MODEL_PATH', "/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

    # Device for inference
    DEVICE: str = "cpu"  # Change to "cuda" if GPU available

    # Use T5 for polishing instead of Qwen (experimental)
    USE_T5: bool = os.getenv('USE_T5', False)

    # ===== DATA PREPROCESSING =====

    # Expected sequence length for model input (MUST match training)
    EXPECTED_FRAMES: int = 60  # DON'T CHANGE without retraining model

    # Add velocity and acceleration features (experimental)
    ADD_FEATURES: bool = False  # Keep False unless model was specifically trained with these features

    # Normalization method:
    # - False: Wrist-centered normalization (original, 126 features, loses spatial position)
    # - True: Hybrid normalization (138 features, preserves hand position + orientation)
    # IMPORTANT: Model must be trained with corresponding normalization method!
    USE_HYBRID_NORMALIZATION: bool = os.getenv('USE_HYBRID_NORMALIZATION', True)  # Suggest: True for better accuracy

    # ========================================================================
    # 4. PERFORMANCE & BUFFER MANAGEMENT
    # ========================================================================

    # ===== BUFFER FLUSH (End of Batch Processing) =====

    # Force emission of buffered words at batch end (prevents losing last word)
    FORCE_FLUSH_ON_BATCH_END: bool = True

    # Minimum frames in buffer before allowing flush (prevents flushing noise)
    MIN_FRAMES_FOR_FLUSH: int = 30

    # Lower confidence threshold for flush (catches uncertain last words that might be valid)
    FLUSH_MIN_CONFIDENCE: float = 0.5  # Lower than MIN_CONFIDENCE_THRESHOLD

    # ========================================================================
    # 5. LOGGING & DEBUGGING
    # ========================================================================

    # ===== DETAILED LOGGING =====

    # Master switch for detailed pipeline logging
    # When enabled, logs all detection steps, model outputs, and decision reasoning
    ENABLE_DETAILED_LOGGING: bool = os.getenv('ENABLE_DETAILED_LOGGING', 'False').lower() in ('true', '1', 'yes')

    # Log level: DEBUG, INFO, WARNING, ERROR
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    # Minimal logging - only detector activity and final predictions
    LOG_MINIMAL: bool = os.getenv('LOG_MINIMAL', 'True').lower() in ('true', '1', 'yes')

    # Log raw model outputs (top-N predictions with probabilities)
    LOG_MODEL_OUTPUTS: bool = os.getenv('LOG_MODEL_OUTPUTS', 'True').lower() in ('true', '1', 'yes')
    LOG_TOP_N_PREDICTIONS: int = int(os.getenv('LOG_TOP_N_PREDICTIONS', 5))

    # Log segmentation details (motion detection, silence frames, etc.)
    LOG_SEGMENTATION: bool = os.getenv('LOG_SEGMENTATION', 'True').lower() in ('true', '1', 'yes')

    # Log sliding window voting mechanism
    LOG_VOTING: bool = os.getenv('LOG_VOTING', 'True').lower() in ('true', '1', 'yes')

    # Log hybrid detector decisions (IoU matching, conflicts, etc.)
    LOG_HYBRID_DECISIONS: bool = os.getenv('LOG_HYBRID_DECISIONS', 'True').lower() in ('true', '1', 'yes')

    # Log confidence filtering decisions
    LOG_FILTERING: bool = os.getenv('LOG_FILTERING', 'True').lower() in ('true', '1', 'yes')

    # Log sentence polishing
    LOG_POLISHING: bool = os.getenv('LOG_POLISHING', 'True').lower() in ('true', '1', 'yes')


settings = Settings()

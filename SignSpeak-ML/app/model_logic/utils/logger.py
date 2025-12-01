"""
Professional Logging System for SignSpeak Pipeline

Provides structured, colorized logging for all pipeline components:
- Segmenter (motion-based word boundary detection)
- Sliding Window (continuous classification)
- Hybrid Detector (combination logic)
- Classifier (model predictions)
- Pipeline Manager (overall flow)
- Polisher (sentence correction)

Enable with ENABLE_DETAILED_LOGGING=True in .env

Author: SignSpeak Team
Date: 2025-01-12
"""

from typing import List, Tuple, Dict, Optional
from datetime import datetime
from .config import settings


class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class PipelineLogger:
    """
    Centralized logging for SignSpeak pipeline.

    Features:
    - Component-specific prefixes and colors
    - Structured output for easy parsing
    - Conditional logging based on settings
    - Timestamps for performance analysis
    """

    # Component colors
    COLORS = {
        'SEGMENTER': Colors.CYAN,
        'SLIDING': Colors.BLUE,
        'HYBRID': Colors.MAGENTA,
        'CLASSIFIER': Colors.GREEN,
        'PIPELINE': Colors.YELLOW,
        'POLISHER': Colors.BRIGHT_MAGENTA,
        'FILTER': Colors.BRIGHT_YELLOW,
        'DEBUG': Colors.BRIGHT_BLACK,
    }

    def __init__(self):
        self.enabled = settings.ENABLE_DETAILED_LOGGING
        self.start_time = datetime.now()

    def _format_timestamp(self) -> str:
        """Get elapsed time since logger creation"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return f"[{elapsed:7.3f}s]"

    def _log(self, component: str, message: str, level: str = "INFO", color: Optional[str] = None):
        """Internal logging method"""
        if not self.enabled:
            return

        if color is None:
            color = self.COLORS.get(component, Colors.WHITE)

        timestamp = self._format_timestamp()
        prefix = f"{color}{component:>10}{Colors.RESET}"
        level_colored = self._colorize_level(level)

        print(f"{timestamp} {level_colored} {prefix} │ {message}")

    def _colorize_level(self, level: str) -> str:
        """Colorize log level"""
        level_colors = {
            'DEBUG': Colors.BRIGHT_BLACK,
            'INFO': Colors.WHITE,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
        }
        color = level_colors.get(level, Colors.WHITE)
        return f"{color}{level:>7}{Colors.RESET}"

    # ========================================================================
    # SEGMENTER LOGGING
    # ========================================================================

    def log_segmenter_frame(self, frame_idx: int, motion: float, is_silence: bool, buffer_size: int):
        """Log frame-by-frame segmenter state"""
        if not settings.LOG_SEGMENTATION:
            return

        state = "SILENCE" if is_silence else "MOTION"
        state_color = Colors.DIM if is_silence else Colors.BOLD
        self._log('SEGMENTER',
                 f"{state_color}Frame {frame_idx:4d}: motion={motion:.4f} state={state} buffer={buffer_size}{Colors.RESET}",
                 level="DEBUG")

    def log_segmenter_burst(self, frame_idx: int, motion: float, threshold: float):
        """Log motion burst detection"""
        if not settings.LOG_SEGMENTATION:
            return

        self._log('SEGMENTER',
                 f"{Colors.BRIGHT_YELLOW}⚡ BURST detected at frame {frame_idx}: motion={motion:.4f} (threshold={threshold:.4f}){Colors.RESET}",
                 level="INFO")

    def log_segmenter_word_detected(self, start_frame: int, end_frame: int, num_frames: int):
        """Log when segmenter detects a word boundary"""
        if not settings.LOG_SEGMENTATION:
            return

        self._log('SEGMENTER',
                 f"{Colors.BRIGHT_GREEN}✓ Word boundary: frames [{start_frame:4d}-{end_frame:4d}] ({num_frames} frames){Colors.RESET}",
                 level="INFO")

    def log_segmenter_alternatives(self, num_variants: int, best_word: str, best_conf: float):
        """Log segmenter alternatives generation"""
        if not settings.LOG_SEGMENTATION:
            return

        self._log('SEGMENTER',
                 f"Generated {num_variants} variants → Best: {best_word} (conf={best_conf:.2%})",
                 level="INFO")

    # ========================================================================
    # SLIDING WINDOW LOGGING
    # ========================================================================

    def log_sliding_window_classification(self, window_idx: int, start_frame: int, end_frame: int,
                                         predicted_word: str, confidence: float):
        """Log individual window classification"""
        if not settings.LOG_VOTING:
            return

        self._log('SLIDING',
                 f"Window #{window_idx:3d} [{start_frame:4d}-{end_frame:4d}]: {predicted_word} (conf={confidence:.2%})",
                 level="DEBUG")

    def log_sliding_voting_state(self, vote_counts: Dict[str, int], threshold: int, total_votes: int):
        """Log voting mechanism state"""
        if not settings.LOG_VOTING:
            return

        # Sort by vote count
        sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        votes_str = ", ".join([f"{word}:{count}" for word, count in sorted_votes[:3]])

        self._log('SLIDING',
                 f"Votes: {votes_str} | Threshold: {threshold}/{total_votes}",
                 level="DEBUG")

    def log_sliding_word_emitted(self, word: str, confidence: float, votes: int, required: int,
                                 start_frame: int, end_frame: int):
        """Log when sliding window emits a word"""
        if not settings.LOG_VOTING:
            return

        self._log('SLIDING',
                 f"{Colors.BRIGHT_GREEN}✓ EMIT: {word} (conf={confidence:.2%}, votes={votes}/{required}, frames=[{start_frame}-{end_frame}]){Colors.RESET}",
                 level="INFO")

    def log_sliding_flush(self, word: str, confidence: float, frames: int):
        """Log sliding window buffer flush"""
        if not settings.LOG_VOTING:
            return

        self._log('SLIDING',
                 f"{Colors.YELLOW}⚠ FLUSH: {word} (conf={confidence:.2%}, frames={frames}){Colors.RESET}",
                 level="INFO")

    # ========================================================================
    # HYBRID DETECTOR LOGGING
    # ========================================================================

    def log_hybrid_input(self, num_seg: int, num_slide: int):
        """Log hybrid detector input"""
        if not settings.LOG_HYBRID_DECISIONS:
            return

        self._log('HYBRID',
                 f"Combining: {num_seg} segmenter + {num_slide} sliding detections",
                 level="INFO")

    def log_hybrid_match(self, seg_word: str, seg_conf: float, seg_frames: Tuple[int, int],
                        slide_word: str, slide_conf: float, slide_frames: Tuple[int, int],
                        iou: float, agreement: bool):
        """Log temporal matching between detectors"""
        if not settings.LOG_HYBRID_DECISIONS:
            return

        match_type = "AGREEMENT" if agreement else "CONFLICT"
        match_color = Colors.BRIGHT_GREEN if agreement else Colors.BRIGHT_RED

        self._log('HYBRID',
                 f"{match_color}{match_type}{Colors.RESET} (IoU={iou:.2f}): "
                 f"Seg[{seg_word}:{seg_conf:.2%}@{seg_frames[0]}-{seg_frames[1]}] vs "
                 f"Slide[{slide_word}:{slide_conf:.2%}@{slide_frames[0]}-{slide_frames[1]}]",
                 level="INFO")

    def log_hybrid_decision(self, chosen_word: str, chosen_conf: float, reason: str,
                          boosted: bool = False):
        """Log hybrid detector decision"""
        if not settings.LOG_HYBRID_DECISIONS:
            return

        boost_indicator = f" {Colors.BRIGHT_GREEN}(+BOOST){Colors.RESET}" if boosted else ""
        self._log('HYBRID',
                 f"→ Selected: {chosen_word} (conf={chosen_conf:.2%}){boost_indicator} | Reason: {reason}",
                 level="INFO")

    def log_hybrid_solo_detection(self, detector: str, word: str, confidence: float,
                                  threshold: float, accepted: bool):
        """Log solo detection (only one detector found it)"""
        if not settings.LOG_HYBRID_DECISIONS:
            return

        status = "ACCEPTED" if accepted else "REJECTED"
        status_color = Colors.BRIGHT_GREEN if accepted else Colors.BRIGHT_RED

        self._log('HYBRID',
                 f"{status_color}Solo {detector}: {word} (conf={confidence:.2%} vs threshold={threshold:.2%}) → {status}{Colors.RESET}",
                 level="INFO")

    def log_hybrid_stats(self, stats: Dict[str, int]):
        """Log hybrid detector statistics"""
        if not settings.LOG_HYBRID_DECISIONS:
            return

        self._log('HYBRID',
                 f"Stats: {stats['segmenter_wins']} seg wins, {stats['sliding_wins']} slide wins, "
                 f"{stats['agreements']} agreements, {stats['conflicts']} conflicts",
                 level="INFO")

    # ========================================================================
    # CLASSIFIER LOGGING
    # ========================================================================

    def log_classifier_input(self, input_shape: Tuple[int, ...], num_features: int):
        """Log classifier input shape"""
        if not settings.LOG_MODEL_OUTPUTS:
            return

        self._log('CLASSIFIER',
                 f"Input: shape={input_shape} features={num_features}",
                 level="DEBUG")

    def log_classifier_raw_output(self, top_predictions: List[Tuple[str, float]]):
        """Log raw model predictions (top-N)"""
        if not settings.LOG_MODEL_OUTPUTS:
            return

        # Format top predictions
        preds_str = ", ".join([f"{word}:{prob:.2%}" for word, prob in top_predictions])

        self._log('CLASSIFIER',
                 f"Raw output (top-{len(top_predictions)}): {preds_str}",
                 level="INFO")

    def log_classifier_prediction(self, word: str, confidence: float, all_probs: Optional[List[float]] = None):
        """Log final classifier prediction"""
        if not settings.LOG_MODEL_OUTPUTS:
            return

        entropy_str = ""
        if all_probs is not None:
            import numpy as np
            # Calculate entropy as uncertainty measure
            probs = np.array(all_probs)
            probs = probs[probs > 0]  # Filter zeros
            entropy = -np.sum(probs * np.log(probs))
            entropy_str = f" entropy={entropy:.2f}"

        self._log('CLASSIFIER',
                 f"→ Prediction: {word} (conf={confidence:.2%}){entropy_str}",
                 level="INFO")

    def log_classifier_batch(self, batch_size: int, predictions: List[str]):
        """Log batch prediction results"""
        if not settings.LOG_MODEL_OUTPUTS:
            return

        unique_words = set(predictions)
        self._log('CLASSIFIER',
                 f"Batch prediction: {batch_size} inputs → {len(unique_words)} unique words",
                 level="INFO")

    # ========================================================================
    # FILTERING LOGGING
    # ========================================================================

    def log_confidence_filter_accept(self, word: str, confidence: float, threshold: float,
                                     source: str, frames: Optional[Tuple[int, int]] = None):
        """Log word accepted by confidence filter"""
        if not settings.LOG_FILTERING:
            return

        frames_str = f" frames=[{frames[0]}-{frames[1]}]" if frames else ""
        self._log('FILTER',
                 f"{Colors.BRIGHT_GREEN}✓ ACCEPT{Colors.RESET} ({source}): {word} (conf={confidence:.2%} ≥ {threshold:.2%}){frames_str}",
                 level="INFO")

    def log_confidence_filter_reject(self, word: str, confidence: float, threshold: float,
                                    source: str, frames: Optional[Tuple[int, int]] = None):
        """Log word rejected by confidence filter"""
        if not settings.LOG_FILTERING:
            return

        frames_str = f" frames=[{frames[0]}-{frames[1]}]" if frames else ""
        self._log('FILTER',
                 f"{Colors.BRIGHT_RED}✗ REJECT{Colors.RESET} ({source}): {word} (conf={confidence:.2%} < {threshold:.2%}){frames_str}",
                 level="INFO")

    def log_special_label_detected(self, label: str, source: str):
        """Log special label (e.g., PUSH) detection"""
        if not settings.LOG_FILTERING:
            return

        self._log('FILTER',
                 f"{Colors.BRIGHT_CYAN}⚡ SPECIAL: {label} detected by {source} → Ending sentence{Colors.RESET}",
                 level="INFO")

    # ========================================================================
    # PIPELINE LOGGING
    # ========================================================================

    def log_pipeline_start(self, num_frames: int, mode: str):
        """Log pipeline processing start"""
        self._log('PIPELINE',
                 f"{Colors.BOLD}▶ START: Processing {num_frames} frames (mode={mode}){Colors.RESET}",
                 level="INFO")

    def log_pipeline_batch_summary(self, num_words: int, words: List[str], processing_time: float):
        """Log batch processing summary"""
        words_str = " → ".join(words) if words else "(no words)"
        self._log('PIPELINE',
                 f"{Colors.BRIGHT_GREEN}◼ BATCH COMPLETE: {num_words} words in {processing_time:.3f}s | {words_str}{Colors.RESET}",
                 level="INFO")

    def log_pipeline_buffer_state(self, word_buffer: List[str], sentence_buffer: List[str]):
        """Log current buffer state"""
        word_str = " ".join(word_buffer) if word_buffer else "(empty)"
        self._log('PIPELINE',
                 f"Buffer: [{word_str}] | Sentences: {len(sentence_buffer)}",
                 level="DEBUG")

    def log_pipeline_flush(self, reason: str, num_frames: int):
        """Log buffer flush operation"""
        self._log('PIPELINE',
                 f"{Colors.YELLOW}⚠ FLUSH: {reason} ({num_frames} frames buffered){Colors.RESET}",
                 level="INFO")

    def log_pipeline_reset(self, buffer_size: int):
        """Log pipeline reset"""
        self._log('PIPELINE',
                 f"{Colors.YELLOW}⟲ RESET: Clearing {buffer_size} buffered words{Colors.RESET}",
                 level="INFO")

    # ========================================================================
    # POLISHER LOGGING
    # ========================================================================

    def log_polisher_input(self, raw_sentence: str, num_words: int):
        """Log sentence polisher input"""
        if not settings.LOG_POLISHING:
            return

        self._log('POLISHER',
                 f"Input: \"{raw_sentence}\" ({num_words} words)",
                 level="INFO")

    def log_polisher_output(self, polished_sentence: str, changed: bool):
        """Log sentence polisher output"""
        if not settings.LOG_POLISHING:
            return

        status = "MODIFIED" if changed else "UNCHANGED"
        status_color = Colors.BRIGHT_YELLOW if changed else Colors.BRIGHT_BLACK

        self._log('POLISHER',
                 f"{status_color}{status}{Colors.RESET}: \"{polished_sentence}\"",
                 level="INFO")

    def log_polisher_model_info(self, model_name: str, inference_time: float):
        """Log polisher model information"""
        if not settings.LOG_POLISHING:
            return

        self._log('POLISHER',
                 f"Model: {model_name} | Inference: {inference_time:.3f}s",
                 level="DEBUG")

    # ========================================================================
    # MINIMAL LOGGING (Simple, clean output)
    # ========================================================================

    def log_minimal(self, component: str, message: str, color: Optional[str] = None):
        """Minimal logging - always shown when LOG_MINIMAL is enabled"""
        if not settings.LOG_MINIMAL:
            return

        if color is None:
            color = self.COLORS.get(component, Colors.WHITE)

        prefix = f"{color}{component:>10}{Colors.RESET}"
        print(f"{prefix} │ {message}")

    def log_detector_start(self, detector_name: str, num_segments: int = None):
        """Log when a detector starts processing"""
        if not settings.LOG_MINIMAL:
            return

        msg = f"{Colors.BOLD}▶ START{Colors.RESET}"
        if num_segments is not None:
            msg += f" - detected {num_segments} segment(s)"
        self.log_minimal(detector_name, msg)

    def log_detector_result(self, detector_name: str, word: str, confidence: float,
                          frames: Tuple[int, int] = None, details: str = None):
        """Log detector result (word detection)"""
        if not settings.LOG_MINIMAL:
            return

        frames_str = f" [{frames[0]}-{frames[1]}]" if frames else ""
        details_str = f" ({details})" if details else ""
        msg = f"→ {word} (conf={confidence:.2%}){frames_str}{details_str}"
        self.log_minimal(detector_name, msg)

    def log_final_prediction(self, word: str, confidence: float, mode: str):
        """Log final prediction added to sentence buffer"""
        if not settings.LOG_MINIMAL:
            return

        msg = f"{Colors.BRIGHT_GREEN}✓ ADDED TO SENTENCE:{Colors.RESET} {word} (conf={confidence:.2%})"
        self.log_minimal(mode, msg)

    def log_detector_flush(self, detector_name: str, word: str, confidence: float):
        """Log detector buffer flush"""
        if not settings.LOG_MINIMAL:
            return

        msg = f"{Colors.YELLOW}⚠ FLUSH:{Colors.RESET} {word} (conf={confidence:.2%})"
        self.log_minimal(detector_name, msg)

    # ========================================================================
    # GENERAL UTILITIES
    # ========================================================================

    def log_separator(self, char: str = "=", width: int = 80):
        """Log visual separator"""
        if self.enabled:
            print(f"{Colors.BRIGHT_BLACK}{char * width}{Colors.RESET}")

    def log_section_header(self, title: str):
        """Log section header"""
        if self.enabled:
            self.log_separator("=")
            print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}  {title.upper()}{Colors.RESET}")
            self.log_separator("=")

    def log_error(self, component: str, error: Exception, context: str = ""):
        """Log error with traceback"""
        error_msg = f"{Colors.BRIGHT_RED}ERROR{Colors.RESET}: {str(error)}"
        if context:
            error_msg = f"{context} | {error_msg}"

        self._log(component, error_msg, level="ERROR")

    def log_warning(self, component: str, message: str):
        """Log warning"""
        self._log(component, f"{Colors.YELLOW}⚠{Colors.RESET} {message}", level="WARNING")

    def log_debug(self, component: str, message: str):
        """Log debug information"""
        self._log(component, message, level="DEBUG")


# Global logger instance
logger = PipelineLogger()


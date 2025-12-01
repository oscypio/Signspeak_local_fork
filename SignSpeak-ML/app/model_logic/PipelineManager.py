from typing import List, Dict, Any
import time

from .polishing.SentencePolisher import SentencePolisher
from .polishing.SentencePolisherT5 import T5Polisher
from .utils.util_functions import *
from .utils.logger import logger
from ..schemas import FrameData
from .preprocessing.DataPreparer import UnifiedDataPreparer
from .segmentation.WordSegmenter import WordSegmenter
from .segmentation.SlidingWindowDetector import SlidingWindowDetector
from .segmentation.HybridDetector import HybridDetector
from .classifier.ASLClassifier import ASLClassifier


class PipelineManager:
    """
    Full ASL processing pipeline:
        FrameData JSON ->
        UnifiedDataPreparer ->
        WordSegmenter ->
        ASLClassifier ->
        Response dict

    This class returns response every time the word is added.
    Also keeps buffer so even when two sentences in one input can be handled.
    """

    def __init__(
        self,
    ):
        # --- Components ---
        self.preparer = UnifiedDataPreparer()
        self.segmenter = WordSegmenter()
        self.sliding_detector = SlidingWindowDetector()
        self.hybrid_detector = HybridDetector()
        self.classifier = ASLClassifier()

        if settings.USE_T5:
            self.polisher = T5Polisher()
        else:
            self.polisher = SentencePolisher()

        # --- Model metadata ---
        self.target_words = self.classifier.class_names

        # --- Sentence buffer ---
        self.word_buffer: List[str] = []
        self.sentence_buffer: List[str] = []

        # --- Global frame counter for unified temporal tracking ---
        # Tracks total frames processed in current session
        # Both detectors will add this offset to their local frame indices
        # to ensure consistent frame numbering across detections
        self.global_frame_counter: int = 0

    # ------------------------------------------------------------------
    # MAIN ENTRYPOINT
    # ------------------------------------------------------------------
    def process(self, frames: List[FrameData]) -> list[Dict[str, Any]]:
        """
        Main processing entry point. Routes to:
        - process_with_hybrid() if USE_HYBRID_MODE is enabled
        - process_with_sliding_window() if USE_SLIDING_WINDOW is enabled
        - process_with_segmenter() (traditional approach) otherwise
        """
        # Determine mode
        if settings.USE_HYBRID_MODE:
            mode = "HYBRID"
        elif settings.USE_SLIDING_WINDOW:
            mode = "SLIDING_WINDOW"
        else:
            mode = "SEGMENTER"

        logger.log_pipeline_start(len(frames), mode)

        if settings.USE_HYBRID_MODE:
            return self.process_with_hybrid(frames)
        elif settings.USE_SLIDING_WINDOW:
            return self.process_with_sliding_window(frames)
        else:
            return self.process_with_segmenter(frames)

    # ------------------------------------------------------------------
    # SLIDING WINDOW PROCESSING
    # ------------------------------------------------------------------
    def process_with_sliding_window(self, frames: List[FrameData]) -> list[Dict[str, Any]]:
        """
        Process frames using sliding window detector.

        This approach doesn't rely on motion-based segmentation.
        Instead, it classifies overlapping temporal windows and emits
        words when stable predictions are detected.
        """
        # ====================================================
        # 1) Preprocess raw API frames
        # ====================================================
        seq = self.preparer.prepare_raw(frames)   # → (T,168)

        if seq is None or len(seq) == 0:
            return [generate_no_word_response()]

        # Store batch size for global frame counter update
        batch_frame_count = len(seq)

        # Set global frame offset in detector for unified frame numbering
        self.sliding_detector.global_frame_offset = self.global_frame_counter

        # ====================================================
        # 2) Process frames through sliding window detector
        # ====================================================
        responses = []
        logger.log_detector_start("SLIDING")

        if settings.SLIDING_WINDOW_BATCH_PREDICT:
            # Optimized batch processing
            detected_words = self.sliding_detector.add_frames_batch_optimized(
                list(seq), self.classifier, self.preparer
            )
        else:
            # Frame-by-frame processing
            detected_words = []
            for vec in seq:
                result = self.sliding_detector.add_frame(
                    vec, self.classifier, self.preparer
                )
                if result is not None:
                    detected_words.append(result)

        # ====================================================
        # 2b) FLUSH BUFFER - Emit last word at batch end
        # ====================================================
        if settings.FORCE_FLUSH_ON_BATCH_END:
            flushed_result = self.sliding_detector.flush_buffer(
                self.classifier, self.preparer
            )
            if flushed_result is not None:
                # flush_buffer now returns 4-tuple: (word, confidence, start_frame, end_frame)
                word, confidence, start_f, end_f = flushed_result
                logger.log_detector_flush("SLIDING", word, confidence)
                detected_words.append((word, confidence, start_f, end_f))

        # ====================================================
        # 3) Process detected words
        # ====================================================
        for detection in detected_words:
            # All detections now return 4-tuple: (word, confidence, start_frame, end_frame)
            word, confidence, start_frame, end_frame = detection

            # ====================================================
            # 4) Special symbol -> end of sentence
            # ====================================================
            if word == settings.SPECIAL_LABEL:
                logger.log_special_label_detected(word, "SLIDING")
                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()

                # Log polishing
                logger.log_polisher_input(final_sentence, len(final_sentence.split()))
                polishing_start = time.time()
                final_sentence = self.polisher.polish(final_sentence)
                polishing_time = time.time() - polishing_start
                logger.log_polisher_output(final_sentence, True)
                logger.log_polisher_model_info("Qwen2.5" if not settings.USE_T5 else "T5", polishing_time)

                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # ====================================================
            # 5) Normal word -> add to buffer (only if confidence above threshold)
            # ====================================================
            if confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
                logger.log_confidence_filter_accept(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "SLIDING", (start_frame, end_frame))
                logger.log_final_prediction(word, confidence, "SLIDING")
                self.word_buffer.append(word)
                responses.append(generate_given_word_response(word, self.word_buffer, confidence))
            else:
                logger.log_confidence_filter_reject(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "SLIDING", (start_frame, end_frame))
                logger.log_detector_result("SLIDING", word, confidence, (start_frame, end_frame), "REJECTED")

        # If no words detected, return empty list (consistency with segmenter)
        if not responses:
            # Still update global frame counter even if no detections
            self.global_frame_counter += batch_frame_count
            return []

        # Update global frame counter for unified temporal tracking
        self.global_frame_counter += batch_frame_count

        return responses

    # ------------------------------------------------------------------
    # TRADITIONAL SEGMENTER PROCESSING
    # ------------------------------------------------------------------
    def process_with_segmenter(self, frames: List[FrameData]) -> list[Dict[str, Any]]:
        """
        Traditional processing pipeline using motion-based segmentation.

        This is the original approach that detects word boundaries
        through motion analysis (silence detection).
        """
        # ====================================================
        # 1) Preprocess raw API frames
        # ====================================================
        seq = self.preparer.prepare_raw(frames)   # → (T,168)

        if seq is None or len(seq) == 0:
            return [generate_no_word_response()]

        # Store batch size for global frame counter update
        batch_frame_count = len(seq)

        # Set global frame offset in detector for unified frame numbering
        self.segmenter.global_frame_offset = self.global_frame_counter

        # ====================================================
        # 2) Feed each frame into segmenter
        # ====================================================
        segments = []  # Will store (segment_data, start_frame, end_frame)
        if settings.USE_SEGMENTATOR:
            logger.log_detector_start("SEGMENTER")

            for vec in seq:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    segs = self.segmenter.add_frame_with_alternatives(vec)
                    if segs is not None:
                        # segs is a list of np.ndarrays (variants for one detected word)
                        # For alternatives mode, we don't have temporal info yet
                        segments.append((segs, None, None))
                else:
                    result = self.segmenter.add_frame(vec)
                    if result is not None:
                        # result is (segment_np, start_frame, end_frame)
                        segments.append(result)

            # ====================================================
            # 2b) FLUSH BUFFER - Emit remaining content at batch end
            # ====================================================
            if settings.FORCE_FLUSH_ON_BATCH_END:
                flushed_result = self.segmenter.flush_buffer()
                if flushed_result is not None:
                    # flushed_result is (segment_np, start_frame, end_frame)
                    flushed_segment, start_f, end_f = flushed_result
                    # Classify the flushed segment
                    prepared = self.preparer.prepare_resampled(flushed_segment)
                    proba_dict = self.classifier.predict_proba(prepared)
                    word = max(proba_dict.items(), key=lambda x: x[1])[0]
                    confidence = proba_dict[word]

                    # Add to segments if confidence acceptable (use stricter threshold)
                    min_conf = max(settings.FLUSH_MIN_CONFIDENCE, settings.MIN_CONFIDENCE_THRESHOLD)
                    if confidence >= min_conf:
                        logger.log_detector_flush("SEGMENTER", word, confidence)
                        # Wrap in list for SEGMENTER_RETURN_ALTERNATIVES compatibility
                        if settings.SEGMENTER_RETURN_ALTERNATIVES:
                            segments.append(([flushed_segment], start_f, end_f))
                        else:
                            segments.append((flushed_segment, start_f, end_f))
        else:
            segments.append((seq, 0, len(seq) - 1))

        logger.log_detector_start("SEGMENTER", len(segments))
        if not segments:
            return []  # Return empty list instead of no_word_response

        # ====================================================
        # 3) Predict label for each detected word segment
        # ====================================================
        responses = []

        for segment_item in segments:
            # Extract segment data and temporal info
            if settings.SEGMENTER_RETURN_ALTERNATIVES:
                segment_data, start_f, end_f = segment_item
                # Delegate selection of best label to classifier
                cand_list = [self.preparer.prepare_resampled(cand) for cand in segment_data]
                word, confidence = self.classifier.predict_best_from_candidates(cand_list, return_confidence=True)
                logger.log_detector_result("SEGMENTER", word, confidence, (start_f, end_f) if start_f is not None else None,
                                         f"{len(segment_data)} alternatives")
            else:
                segment_np, start_f, end_f = segment_item
                # 1. generate TTA variants
                tta_variants = self.preparer.prepare_tta_segments(segment_np, n_augs=7)
                # 2. predict using majority vote
                word, confidence = self.classifier.predict_tta(tta_variants, return_confidence=True)
                logger.log_detector_result("SEGMENTER", word, confidence, (start_f, end_f))

            # ====================================================
            # 4) Special symbol -> end of sentence
            # ====================================================
            if word == settings.SPECIAL_LABEL:
                logger.log_special_label_detected(word, "SEGMENTER")
                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()
                final_sentence = self.polisher.polish(final_sentence)
                self.segmenter.reset()
                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # ====================================================
            # 5) Normal word -> add to buffer (only if confidence above threshold)
            # ====================================================
            if confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
                logger.log_confidence_filter_accept(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "SEGMENTER", (start_f, end_f) if start_f is not None else None)
                logger.log_final_prediction(word, confidence, "SEGMENTER")
                self.word_buffer.append(word)
                responses.append(generate_given_word_response(word, self.word_buffer, confidence))
            else:
                logger.log_confidence_filter_reject(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "SEGMENTER", (start_f, end_f) if start_f is not None else None)
                logger.log_detector_result("SEGMENTER", word, confidence, (start_f, end_f) if start_f is not None else None, "REJECTED")
                responses.append(generate_no_word_response('Low confidence word ignored'))

        # Update global frame counter for unified temporal tracking
        self.global_frame_counter += batch_frame_count

        # Return responses (can be empty list if no words above threshold)
        return_val = responses if settings.USE_SEGMENTATOR else ([responses[-1]] if responses else [])
        return return_val

    # ------------------------------------------------------------------
    # HYBRID PROCESSING
    # ------------------------------------------------------------------
    def process_with_hybrid(self, frames: List[FrameData]) -> list[Dict[str, Any]]:
        """
        Hybrid processing pipeline using both sliding window and segmentation.

        Runs both detectors in parallel and combines results using HybridDetector
        for improved accuracy by leveraging strengths of both approaches.
        """
        # ====================================================
        # 1) Preprocess raw API frames
        # ====================================================
        seq = self.preparer.prepare_raw(frames)   # → (T,168)

        if seq is None or len(seq) == 0:
            return [generate_no_word_response()]

        # Store batch size for global frame counter update
        batch_frame_count = len(seq)

        # Set global frame offset in detectors for unified frame numbering
        self.sliding_detector.global_frame_offset = self.global_frame_counter
        self.segmenter.global_frame_offset = self.global_frame_counter

        # ====================================================
        # 2a) Run sliding window detector
        # ====================================================
        logger.log_detector_start("SLIDING")
        sliding_results = []  # Will store (word, confidence, start_frame, end_frame)
        if settings.SLIDING_WINDOW_BATCH_PREDICT:
            sliding_results = self.sliding_detector.add_frames_batch_optimized(
                list(seq), self.classifier, self.preparer
            )
        else:
            for vec in seq:
                result = self.sliding_detector.add_frame(
                    vec, self.classifier, self.preparer
                )
                if result is not None:
                    sliding_results.append(result)

        # ====================================================
        # 2b) Run traditional segmenter
        # ====================================================
        logger.log_detector_start("SEGMENTER")
        segmenter_results = []  # Will store (word, confidence, start_frame, end_frame)
        segments = []  # Will store segment data with temporal info

        if settings.USE_SEGMENTATOR:
            for vec in seq:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    segs = self.segmenter.add_frame_with_alternatives(vec)
                    if segs is not None:
                        # For alternatives, we don't have temporal info in this mode
                        segments.append((segs, None, None))
                else:
                    result = self.segmenter.add_frame(vec)
                    if result is not None:
                        # result is (segment_np, start_frame, end_frame)
                        segments.append(result)

            # Classify segmenter detections
            for segment_item in segments:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    segment_data, start_f, end_f = segment_item
                    # Use alternatives for better accuracy
                    cand_list = [self.preparer.prepare_resampled(cand) for cand in segment_data]
                    word = self.classifier.predict_best_from_candidates(cand_list)
                    # Get confidence from best candidate
                    proba_dicts = self.classifier.predict_proba_batch(cand_list)
                    confidences = [max(proba.values()) for proba in proba_dicts]
                    confidence = max(confidences)
                    # Use placeholder temporal info for alternatives mode
                    logger.log_detector_result("SEGMENTER", word, confidence, (0, len(seq) - 1),
                                             f"{len(segment_data)} alternatives")
                    segmenter_results.append((word, confidence, 0, len(seq) - 1))
                else:
                    # Standard TTA approach with temporal info
                    segment_np, start_f, end_f = segment_item
                    tta_variants = self.preparer.prepare_tta_segments(segment_np, n_augs=7)
                    word = self.classifier.predict_tta(tta_variants)
                    # Get confidence from TTA
                    proba_dict = self.classifier.predict_proba(self.preparer.prepare_resampled(segment_np))
                    confidence = proba_dict.get(word, 0.5)
                    logger.log_detector_result("SEGMENTER", word, confidence, (start_f, end_f))
                    segmenter_results.append((word, confidence, start_f, end_f))

        # ====================================================
        # 2c) FLUSH BUFFERS - Emit remaining content at batch end
        # ====================================================
        if settings.FORCE_FLUSH_ON_BATCH_END:
            # Flush segmenter buffer
            if settings.USE_SEGMENTATOR:
                flushed_result = self.segmenter.flush_buffer()
                if flushed_result is not None:
                    flushed_segment, start_f, end_f = flushed_result
                    # Classify the flushed segment (unified code - no duplication)
                    prepared = self.preparer.prepare_resampled(flushed_segment)
                    word = self.classifier.predict_label(prepared)
                    proba_dict = self.classifier.predict_proba(prepared)
                    confidence = proba_dict.get(word, 0.5)

                    # Add to segmenter results if confidence is acceptable
                    if confidence >= settings.FLUSH_MIN_CONFIDENCE:
                        logger.log_detector_flush("SEGMENTER", word, confidence)
                        segmenter_results.append((word, confidence, start_f, end_f))

            # Flush sliding window buffer
            flushed_sliding = self.sliding_detector.flush_buffer(
                self.classifier, self.preparer
            )
            if flushed_sliding is not None:
                # flush_buffer now returns 4-tuple: (word, confidence, start_frame, end_frame)
                word, confidence, start_f, end_f = flushed_sliding
                logger.log_detector_flush("SLIDING", word, confidence)
                sliding_results.append((word, confidence, start_f, end_f))

        # ====================================================
        # 3) Combine results using hybrid detector
        # ====================================================
        logger.log_detector_start("HYBRID")
        combined_results = self.hybrid_detector.combine_detections(
            segmenter_results, sliding_results
        )

        # ====================================================
        # 3b) Sort combined results by start frame to preserve temporal order
        # ====================================================
        combined_results.sort(key=lambda x: x[2])  # Sort by start_frame (index 2)

        # ====================================================
        # 4) Process combined detections
        # ====================================================
        responses = []
        start_time = time.time()

        for word, confidence, start_f, end_f in combined_results:
            # Special symbol -> end of sentence
            if word == settings.SPECIAL_LABEL:
                logger.log_special_label_detected(word, "HYBRID")
                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()

                # Log polishing
                logger.log_polisher_input(final_sentence, len(final_sentence.split()))
                polishing_start = time.time()
                final_sentence = self.polisher.polish(final_sentence)
                polishing_time = time.time() - polishing_start
                logger.log_polisher_output(final_sentence, True)
                logger.log_polisher_model_info("Qwen2.5" if not settings.USE_T5 else "T5", polishing_time)

                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # Normal word -> add to buffer (only if confidence above threshold)
            if confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
                logger.log_confidence_filter_accept(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "HYBRID", (start_f, end_f))
                logger.log_final_prediction(word, confidence, "HYBRID")
                self.word_buffer.append(word)
                responses.append(generate_given_word_response(word, self.word_buffer, confidence))
            else:
                logger.log_confidence_filter_reject(word, confidence, settings.MIN_CONFIDENCE_THRESHOLD,
                                                   "HYBRID", (start_f, end_f))
                logger.log_detector_result("HYBRID", word, confidence, (start_f, end_f), "REJECTED")

        # If no words detected, return no-word response
        if not responses:
            return [generate_no_word_response()]

        # Log batch summary
        processing_time = time.time() - start_time
        detected_words_list = [resp.get('prediction', '') for resp in responses if resp.get('prediction')]
        logger.log_pipeline_batch_summary(len(detected_words_list), detected_words_list, processing_time)
        logger.log_pipeline_buffer_state(self.word_buffer, self.sentence_buffer)

        # Update global frame counter for unified temporal tracking
        self.global_frame_counter += batch_frame_count

        # NO RESET HERE - let buffer persist for next batch
        return responses

    def force_end_sentence(self) -> Dict[str, Any]:
        """Force sentence finalization as if SPECIAL_LABEL was predicted."""
        final_sentence = " ".join(self.word_buffer)
        self.reset_buffer()
        # Polish sentence
        final_sentence = self.polisher.polish(final_sentence)
        self.sentence_buffer.append(final_sentence)
        response = generate_end_of_sentence_response(final_sentence)
        return response

    def reset_buffer(self):
        """
        Reset internal word buffer, detector state, and global frame counter.

        This should be called between independent sessions to:
        - Clear accumulated words
        - Reset detector internal buffers
        - Reset frame counter to prevent overflow
        """
        self.word_buffer.clear()
        if settings.USE_SEGMENTATOR:
            self.segmenter.reset()
        if settings.USE_SLIDING_WINDOW:
            self.sliding_detector.reset()
        # Reset global frame counter to prevent overflow and start fresh session
        self.global_frame_counter = 0

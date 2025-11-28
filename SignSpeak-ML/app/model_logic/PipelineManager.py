from typing import List, Dict, Any

from .polishing.SentencePolisher import SentencePolisher
from .polishing.SentencePolisherT5 import T5Polisher
from .utils.util_functions import *
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

        # ====================================================
        # 2) Process frames through sliding window detector
        # ====================================================
        responses = []

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
        # 3) Process detected words
        # ====================================================
        for word, confidence in detected_words:

            # ====================================================
            # 4) Special symbol -> end of sentence
            # ====================================================
            if word == settings.SPECIAL_LABEL:
                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()
                final_sentence = self.polisher.polish(final_sentence)

                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # ====================================================
            # 5) Normal word -> add to buffer
            # ====================================================
            self.word_buffer.append(word)
            responses.append(generate_given_word_response(word, self.word_buffer))

        # If no words detected, return no-word response
        if not responses:
            return [generate_no_word_response()]

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

        # ====================================================
        # 2) Feed each frame into segmenter
        # ====================================================
        segments = []
        if settings.USE_SEGMENTATOR:

            for vec in seq:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    segs = self.segmenter.add_frame_with_alternatives(vec)
                    if segs is not None:
                        # segs is a list of np.ndarrays (variants for one detected word)
                        segments.append(segs)
                else:
                    segment = self.segmenter.add_frame(vec)
                    if segment is not None:
                        segments.append(segment)

            if not segments:
                print("No segments found")
                return [generate_no_word_response()]

            print(f"Detected {len(segments)} segments.")

            # ====================================================
            # 2b) FLUSH BUFFER - Emit remaining content at batch end
            # ====================================================
            if settings.FORCE_FLUSH_ON_BATCH_END:
                flushed_segment = self.segmenter.flush_buffer()
                if flushed_segment is not None:
                    # Classify the flushed segment
                    prepared = self.preparer.prepare_resampled(flushed_segment)
                    proba_dict = self.classifier.predict_proba(prepared)
                    word = max(proba_dict.items(), key=lambda x: x[1])[0]
                    confidence = proba_dict[word]

                    # Add to segments if confidence acceptable
                    if confidence >= settings.FLUSH_MIN_CONFIDENCE:
                        print(f"[FLUSH] Emitting buffered segment: {word} (conf={confidence:.2f})")
                        # Wrap in list for SEGMENTER_RETURN_ALTERNATIVES compatibility
                        if settings.SEGMENTER_RETURN_ALTERNATIVES:
                            segments.append([flushed_segment])
                        else:
                            segments.append(flushed_segment)
        else:
            segments.append(seq)

        # ====================================================
        # 3) Predict label for each detected word segment
        # ====================================================
        responses = []

        for segment_item in segments:

            # If using alternative segments, segment_item is list of variants
            if settings.USE_SEGMENTATOR and settings.SEGMENTER_RETURN_ALTERNATIVES:

                # Delegate selection of best label to classifier
                cand_list = [self.preparer.prepare_resampled(cand) for cand in segment_item]

                word = self.classifier.predict_best_from_candidates(cand_list)

            else:
                segment_np = segment_item

                # 1. generate TTA variants
                tta_variants = self.preparer.prepare_tta_segments(segment_np, n_augs=7)

                # 2. predict using majority vote
                word = self.classifier.predict_tta(tta_variants)

            # ====================================================
            # 4) Special symbol -> end of sentence
            # ====================================================
            if word == settings.SPECIAL_LABEL:

                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()
                final_sentence = self.polisher.polish(final_sentence)

                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # ====================================================
            # 5) Normal word -> add to buffer
            # ====================================================
            self.word_buffer.append(word)

            responses.append(generate_given_word_response(word, self.word_buffer))
        self.segmenter.reset()
        return_val = responses if settings.USE_SEGMENTATOR else [responses[-1]]
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

        # ====================================================
        # 2a) Run sliding window detector
        # ====================================================
        sliding_results = []
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
        segmenter_results = []
        segments = []

        if settings.USE_SEGMENTATOR:
            for vec in seq:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    segs = self.segmenter.add_frame_with_alternatives(vec)
                    if segs is not None:
                        segments.append(segs)
                else:
                    segment = self.segmenter.add_frame(vec)
                    if segment is not None:
                        segments.append(segment)

            # Classify segmenter detections
            for segment_item in segments:
                if settings.SEGMENTER_RETURN_ALTERNATIVES:
                    # Use alternatives for better accuracy
                    cand_list = [self.preparer.prepare_resampled(cand) for cand in segment_item]
                    word = self.classifier.predict_best_from_candidates(cand_list)
                    # Get confidence from best candidate
                    proba_dicts = self.classifier.predict_proba_batch(cand_list)
                    confidences = [max(proba.values()) for proba in proba_dicts]
                    confidence = max(confidences)
                else:
                    # Standard TTA approach
                    segment_np = segment_item
                    tta_variants = self.preparer.prepare_tta_segments(segment_np, n_augs=7)
                    word = self.classifier.predict_tta(tta_variants)
                    # Get confidence from TTA
                    proba_dict = self.classifier.predict_proba(self.preparer.prepare_resampled(segment_np))
                    confidence = proba_dict.get(word, 0.5)

                segmenter_results.append((word, confidence))

        # ====================================================
        # 2c) FLUSH BUFFERS - Emit remaining content at batch end
        # ====================================================
        if settings.FORCE_FLUSH_ON_BATCH_END:
            # Flush segmenter buffer
            if settings.USE_SEGMENTATOR:
                flushed_segment = self.segmenter.flush_buffer()
                if flushed_segment is not None:
                    # Classify the flushed segment
                    if settings.SEGMENTER_RETURN_ALTERNATIVES:
                        # Generate alternatives for flushed content
                        prepared = self.preparer.prepare_resampled(flushed_segment)
                        word = self.classifier.predict_label(prepared)
                        proba_dict = self.classifier.predict_proba(prepared)
                        confidence = proba_dict.get(word, 0.5)
                    else:
                        prepared = self.preparer.prepare_resampled(flushed_segment)
                        word = self.classifier.predict_label(prepared)
                        proba_dict = self.classifier.predict_proba(prepared)
                        confidence = proba_dict.get(word, 0.5)

                    # Add to segmenter results if confidence is acceptable
                    if confidence >= settings.FLUSH_MIN_CONFIDENCE:
                        segmenter_results.append((word, confidence))

            # Flush sliding window buffer
            flushed_sliding = self.sliding_detector.flush_buffer(
                self.classifier, self.preparer
            )
            if flushed_sliding is not None:
                sliding_results.append(flushed_sliding)

        # ====================================================
        # 3) Combine results using hybrid detector
        # ====================================================
        combined_results = self.hybrid_detector.combine_detections(
            segmenter_results, sliding_results
        )

        # ====================================================
        # 4) Process combined detections
        # ====================================================
        responses = []

        for word, confidence in combined_results:
            # Special symbol -> end of sentence
            if word == settings.SPECIAL_LABEL:
                final_sentence = " ".join(self.word_buffer)
                self.word_buffer.clear()
                final_sentence = self.polisher.polish(final_sentence)
                self.sentence_buffer.append(final_sentence)
                responses.append(generate_end_of_sentence_response(final_sentence))
                continue

            # Normal word -> add to buffer
            self.word_buffer.append(word)
            responses.append(generate_given_word_response(word, self.word_buffer))

        # If no words detected, return no-word response
        if not responses:
            return [generate_no_word_response()]
        self.segmenter.reset()
        return responses

    def force_end_sentence(self) -> Dict[str, Any]:
        """Force sentence finalization as if SPECIAL_LABEL was predicted."""
        final_sentence = " ".join(self.word_buffer)
        self.word_buffer.clear()
        # Polish sentence
        final_sentence = self.polisher.polish(final_sentence)
        self.sentence_buffer.append(final_sentence)
        response = generate_end_of_sentence_response(final_sentence)
        return response

    def reset_buffer(self):
        """Reset internal word buffer and detector state."""
        self.word_buffer.clear()
        if settings.USE_SLIDING_WINDOW:
            self.sliding_detector.reset()

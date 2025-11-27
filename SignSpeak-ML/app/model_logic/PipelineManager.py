from typing import List, Dict, Any

from .polishing.SentencePolisher import SentencePolisher
from .polishing.SentencePolisherT5 import T5Polisher
from .utils.util_functions import *
from ..schemas import FrameData
from .preprocessing.DataPreparer import UnifiedDataPreparer
from .segmentation.WordSegmenter import WordSegmenter
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

        return_val = responses if settings.USE_SEGMENTATOR else [responses[-1]]
        return return_val

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
        """Reset internal word buffer."""
        self.word_buffer.clear()

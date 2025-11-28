"""
Hybrid Detector - Combines Segmenter and Sliding Window

This module intelligently combines traditional motion-based segmentation
with sliding window detection to achieve better accuracy by leveraging
the strengths of both approaches.

Key Features:
- Runs both detectors in parallel
- Combines results using configurable strategies
- Boosts confidence when both methods agree
- Falls back to stronger method when one fails

Author: SignSpeak Team
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..utils.config import settings


class HybridDetector:
    """
    Hybrid detection combining traditional segmenter and sliding window.

    Strategies:
    - 'max_confidence': Choose the detection with highest confidence
    - 'voting': If both detect same word, use it; otherwise use max confidence
    - 'segmenter_primary': Use segmenter unless its confidence is too low
    - 'sliding_primary': Use sliding window unless its confidence is too low

    Parameters:
        strategy: How to combine results (default from config)
        confidence_threshold: Min difference to prefer one over another
        agreement_boost: Confidence boost when both methods agree
    """

    def __init__(
        self,
        strategy: str = settings.HYBRID_STRATEGY,
        confidence_threshold: float = settings.HYBRID_CONFIDENCE_THRESHOLD,
        agreement_boost: float = settings.HYBRID_AGREEMENT_BOOST,
    ):
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.agreement_boost = agreement_boost

        # Statistics for debugging/monitoring
        self.stats = {
            'total_detections': 0,
            'segmenter_wins': 0,
            'sliding_wins': 0,
            'agreements': 0,
            'disagreements': 0,
            'conflicts': 0,  # When same temporal region but different words
            'total_boosted': 0,  # Times agreement boost was applied
        }

    def combine_detections(
        self,
        segmenter_results: List[Tuple[str, float]],
        sliding_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Combine detections from both methods using configured strategy.

        Args:
            segmenter_results: List of (word, confidence) from segmenter
            sliding_results: List of (word, confidence) from sliding window

        Returns:
            Combined list of (word, confidence) tuples
        """
        if not segmenter_results and not sliding_results:
            return []

        if not segmenter_results:
            return sliding_results

        if not sliding_results:
            return segmenter_results

        # Apply strategy
        if self.strategy == 'max_confidence':
            return self._combine_max_confidence(segmenter_results, sliding_results)
        elif self.strategy == 'voting':
            return self._combine_voting(segmenter_results, sliding_results)
        elif self.strategy == 'segmenter_primary':
            return self._combine_segmenter_primary(segmenter_results, sliding_results)
        elif self.strategy == 'sliding_primary':
            return self._combine_sliding_primary(segmenter_results, sliding_results)
        else:
            # Default to max_confidence
            return self._combine_max_confidence(segmenter_results, sliding_results)

    def _combine_max_confidence(
        self,
        segmenter_results: List[Tuple[str, float]],
        sliding_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Choose detection with highest confidence from either method.
        Boost confidence if both methods agree.
        """
        combined = []

        # Create lookup maps for faster access
        seg_map = {word: conf for word, conf in segmenter_results}
        slide_map = {word: conf for word, conf in sliding_results}

        # Get all unique words detected
        all_words = set(seg_map.keys()) | set(slide_map.keys())

        for word in all_words:
            seg_conf = seg_map.get(word, 0.0)
            slide_conf = slide_map.get(word, 0.0)

            if seg_conf > 0 and slide_conf > 0:
                # Both methods detected this word - AGREEMENT!
                self.stats['agreements'] += 1
                self.stats['total_boosted'] += 1

                # Use max confidence and apply boost
                max_conf = max(seg_conf, slide_conf)
                boosted_conf = min(1.0, max_conf + self.agreement_boost)
                combined.append((word, boosted_conf))

            elif seg_conf > slide_conf:
                # Only segmenter or segmenter has higher confidence
                self.stats['segmenter_wins'] += 1
                combined.append((word, seg_conf))

            else:
                # Only sliding window or sliding window has higher confidence
                self.stats['sliding_wins'] += 1
                combined.append((word, slide_conf))

        self.stats['total_detections'] += len(combined)

        # Apply word-level deduplication
        combined = self._deduplicate_by_word(combined)

        return combined

    def _combine_voting(
        self,
        segmenter_results: List[Tuple[str, float]],
        sliding_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Use majority voting: if both agree, use that word.
        If they disagree, include both but mark disagreements.
        """
        combined = []

        seg_map = {word: conf for word, conf in segmenter_results}
        slide_map = {word: conf for word, conf in sliding_results}

        # Find agreements
        agreements = set(seg_map.keys()) & set(slide_map.keys())

        for word in agreements:
            self.stats['agreements'] += 1
            # Average confidence with boost for agreement
            avg_conf = (seg_map[word] + slide_map[word]) / 2
            boosted_conf = min(1.0, avg_conf + self.agreement_boost)
            combined.append((word, boosted_conf))

        # For disagreements, include the one with higher confidence
        # Only if confidence difference is significant
        seg_only = set(seg_map.keys()) - agreements
        slide_only = set(slide_map.keys()) - agreements

        for word in seg_only:
            # Check if there's a competing word from sliding window
            if slide_only:
                max_slide_conf = max(slide_map.get(w, 0) for w in slide_only)
                if seg_map[word] > max_slide_conf + self.confidence_threshold:
                    self.stats['segmenter_wins'] += 1
                    combined.append((word, seg_map[word]))
                else:
                    self.stats['disagreements'] += 1
            else:
                self.stats['segmenter_wins'] += 1
                combined.append((word, seg_map[word]))

        for word in slide_only:
            # Check if there's a competing word from segmenter
            if seg_only:
                max_seg_conf = max(seg_map.get(w, 0) for w in seg_only)
                if slide_map[word] > max_seg_conf + self.confidence_threshold:
                    self.stats['sliding_wins'] += 1
                    combined.append((word, slide_map[word]))
                else:
                    self.stats['disagreements'] += 1
            else:
                self.stats['sliding_wins'] += 1
                combined.append((word, slide_map[word]))

        self.stats['total_detections'] += len(combined)
        return combined

    def _combine_segmenter_primary(
        self,
        segmenter_results: List[Tuple[str, float]],
        sliding_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Prefer segmenter results unless confidence is too low,
        then fall back to sliding window.
        """
        combined = []

        for word, conf in segmenter_results:
            if conf >= settings.SLIDING_WINDOW_MIN_CONFIDENCE:
                self.stats['segmenter_wins'] += 1
                combined.append((word, conf))
            else:
                # Segmenter confidence too low, check sliding window
                slide_detection = next(
                    ((w, c) for w, c in sliding_results if w == word),
                    None
                )
                if slide_detection:
                    self.stats['agreements'] += 1
                    # Use sliding window confidence with small boost
                    boosted_conf = min(1.0, slide_detection[1] + 0.05)
                    combined.append((word, boosted_conf))

        # Add sliding window detections not in segmenter results
        seg_words = {word for word, _ in segmenter_results}
        for word, conf in sliding_results:
            if word not in seg_words:
                self.stats['sliding_wins'] += 1
                combined.append((word, conf))

        self.stats['total_detections'] += len(combined)
        return combined

    def _combine_sliding_primary(
        self,
        segmenter_results: List[Tuple[str, float]],
        sliding_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Prefer sliding window results unless confidence is too low,
        then fall back to segmenter.
        """
        combined = []

        for word, conf in sliding_results:
            if conf >= settings.SLIDING_WINDOW_MIN_CONFIDENCE:
                self.stats['sliding_wins'] += 1
                combined.append((word, conf))
            else:
                # Sliding confidence too low, check segmenter
                seg_detection = next(
                    ((w, c) for w, c in segmenter_results if w == word),
                    None
                )
                if seg_detection:
                    self.stats['agreements'] += 1
                    # Use segmenter confidence with small boost
                    boosted_conf = min(1.0, seg_detection[1] + 0.05)
                    combined.append((word, boosted_conf))

        # Add segmenter detections not in sliding window results
        slide_words = {word for word, _ in sliding_results}
        for word, conf in segmenter_results:
            if word not in slide_words:
                self.stats['segmenter_wins'] += 1
                combined.append((word, conf))

        self.stats['total_detections'] += len(combined)
        return combined

    def _calculate_temporal_iou(
        self,
        start1: int, end1: int,
        start2: int, end2: int
    ) -> float:
        """
        Calculate Intersection over Union for temporal segments.

        Args:
            start1, end1: First segment boundaries (frame indices)
            start2, end2: Second segment boundaries (frame indices)

        Returns:
            float: IoU in range [0, 1]
        """
        # Intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)

        # Union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start

        if union == 0:
            return 0.0

        return intersection / union

    def _deduplicate_by_word(
        self,
        detections: List[Tuple[str, float]],
        strategy: str = None
    ) -> List[Tuple[str, float]]:
        """
        Remove duplicate detections of same word.

        Args:
            detections: List of (word, confidence) tuples
            strategy: Deduplication strategy ('max_confidence', 'first', 'merge')

        Returns:
            Deduplicated list of (word, confidence) tuples
        """
        if not settings.HYBRID_WORD_DEDUP_ENABLED:
            return detections

        if len(detections) <= 1:
            return detections

        strategy = strategy or settings.HYBRID_DEDUP_STRATEGY

        deduplicated = []
        seen_words = {}  # word -> (confidence, index)

        for i, (word, conf) in enumerate(detections):
            if word in seen_words:
                # Same word already seen
                prev_conf, prev_idx = seen_words[word]

                if strategy == 'max_confidence':
                    # Keep detection with higher confidence
                    if conf > prev_conf:
                        # Replace previous detection
                        deduplicated[prev_idx] = (word, conf)
                        seen_words[word] = (conf, prev_idx)
                    # else: keep previous (higher confidence)

                elif strategy == 'first':
                    # Keep first detection (do nothing)
                    pass

                elif strategy == 'merge':
                    # Average confidences
                    merged_conf = (conf + prev_conf) / 2.0
                    deduplicated[prev_idx] = (word, merged_conf)
                    seen_words[word] = (merged_conf, prev_idx)

            else:
                # New word
                seen_words[word] = (conf, len(deduplicated))
                deduplicated.append((word, conf))

        return deduplicated

    def get_statistics(self) -> Dict[str, Any]:
        """Return detection statistics for monitoring/debugging."""
        total = self.stats['total_detections']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'segmenter_win_rate': self.stats['segmenter_wins'] / total,
            'sliding_win_rate': self.stats['sliding_wins'] / total,
            'agreement_rate': self.stats['agreements'] / total,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'total_detections': 0,
            'segmenter_wins': 0,
            'sliding_wins': 0,
            'agreements': 0,
            'disagreements': 0,
        }

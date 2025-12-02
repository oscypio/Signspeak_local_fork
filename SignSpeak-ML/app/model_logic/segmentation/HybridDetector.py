"""
Hybrid Detector - Combines Segmenter and Sliding Window

This module intelligently combines traditional motion-based segmentation
with sliding window detection to achieve better accuracy by leveraging
the strengths of both approaches.

"""

from typing import List, Tuple, Dict, Any
from ..utils.config import settings
from ..utils.logger import logger


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
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Combine detections from both methods using configured strategy.

        Args:
            segmenter_results: List of (word, confidence, start_frame, end_frame) from segmenter
            sliding_results: List of (word, confidence, start_frame, end_frame) from sliding window

        Returns:
            Combined list of (word, confidence, start_frame, end_frame) tuples sorted by start_frame
        """
        # Log input
        logger.log_hybrid_input(len(segmenter_results), len(sliding_results))

        if not segmenter_results and not sliding_results:
            return []

        if not segmenter_results:
            logger.log_debug('HYBRID', 'No segmenter results, using sliding only')
            return sorted(sliding_results, key=lambda x: x[2])

        if not sliding_results:
            logger.log_debug('HYBRID', 'No sliding results, using segmenter only')
            return sorted(segmenter_results, key=lambda x: x[2])

        # Apply strategy
        if self.strategy == 'adaptive':
            return self._combine_adaptive(segmenter_results, sliding_results)
        elif self.strategy == 'max_confidence':
            return self._combine_max_confidence(segmenter_results, sliding_results)
        elif self.strategy == 'voting':
            return self._combine_voting(segmenter_results, sliding_results)
        elif self.strategy == 'segmenter_primary':
            return self._combine_segmenter_primary(segmenter_results, sliding_results)
        elif self.strategy == 'sliding_primary':
            return self._combine_sliding_primary(segmenter_results, sliding_results)
        else:
            # Default to adaptive
            return self._combine_adaptive(segmenter_results, sliding_results)

    def _combine_max_confidence(
        self,
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Choose detection with highest confidence from either method.
        Boost confidence if both methods agree (same word + overlapping temporal region).
        """
        combined = []
        used_seg_indices = set()
        used_slide_indices = set()

        # For each segmenter detection, find best matching sliding detection
        for seg_idx, (seg_word, seg_conf, seg_start, seg_end) in enumerate(segmenter_results):
            best_match = None
            best_iou = 0.0
            best_slide_idx = -1

            # Find sliding detections with temporal overlap
            for slide_idx, (slide_word, slide_conf, slide_start, slide_end) in enumerate(sliding_results):
                if slide_idx in used_slide_indices:
                    continue

                # Calculate temporal IoU
                iou = self._calculate_temporal_iou(seg_start, seg_end, slide_start, slide_end)

                # Check if there's significant overlap and same word
                if iou > settings.HYBRID_OVERLAP_THRESHOLD and seg_word == slide_word:
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (slide_word, slide_conf, slide_start, slide_end)
                        best_slide_idx = slide_idx

            if best_match:
                # AGREEMENT: Both methods detected same word in overlapping region
                self.stats['agreements'] += 1
                self.stats['total_boosted'] += 1

                # Use max confidence and apply agreement boost
                max_conf = max(seg_conf, best_match[1])
                boosted_conf = min(1.0, max_conf + self.agreement_boost)

                # Use temporal boundaries from higher-confidence detection
                if seg_conf > best_match[1]:
                    combined.append((seg_word, boosted_conf, seg_start, seg_end))
                    self.stats['segmenter_wins'] += 1
                else:
                    combined.append((best_match[0], boosted_conf, best_match[2], best_match[3]))
                    self.stats['sliding_wins'] += 1

                used_seg_indices.add(seg_idx)
                used_slide_indices.add(best_slide_idx)
            else:
                # Only segmenter detected this word
                if seg_conf > 0:  # FIX: Check confidence is actually > 0
                    self.stats['segmenter_wins'] += 1
                    combined.append((seg_word, seg_conf, seg_start, seg_end))
                    used_seg_indices.add(seg_idx)

        # Add sliding-only detections that weren't matched
        for slide_idx, (slide_word, slide_conf, slide_start, slide_end) in enumerate(sliding_results):
            if slide_idx not in used_slide_indices:
                if slide_conf > 0:  # FIX: Check confidence is actually > 0
                    self.stats['sliding_wins'] += 1
                    combined.append((slide_word, slide_conf, slide_start, slide_end))

        self.stats['total_detections'] += len(combined)

        # Sort by start frame to preserve temporal order
        combined.sort(key=lambda x: x[2])

        # Apply word-level deduplication with temporal awareness
        if settings.HYBRID_WORD_DEDUP_ENABLED:
            combined = self._deduplicate_by_word_with_temporal(combined)

        return combined

    def _combine_voting(
        self,
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Use majority voting: if both agree, use that word.
        If they disagree, include both but mark disagreements.

        NOTE: This strategy doesn't use temporal information optimally.
        Consider using 'max_confidence' strategy instead for better temporal awareness.
        """
        combined = []

        # Build maps with temporal info
        seg_map = {word: (conf, start, end) for word, conf, start, end in segmenter_results}
        slide_map = {word: (conf, start, end) for word, conf, start, end in sliding_results}

        # Find agreements
        agreements = set(seg_map.keys()) & set(slide_map.keys())

        for word in agreements:
            self.stats['agreements'] += 1
            seg_conf, seg_start, seg_end = seg_map[word]
            slide_conf, slide_start, slide_end = slide_map[word]
            # Average confidence with boost for agreement
            avg_conf = (seg_conf + slide_conf) / 2
            boosted_conf = min(1.0, avg_conf + self.agreement_boost)
            # Use temporal boundaries from higher-confidence detection
            if seg_conf > slide_conf:
                combined.append((word, boosted_conf, seg_start, seg_end))
            else:
                combined.append((word, boosted_conf, slide_start, slide_end))

        # For disagreements, include the one with higher confidence
        seg_only = set(seg_map.keys()) - agreements
        slide_only = set(slide_map.keys()) - agreements

        for word in seg_only:
            seg_conf, seg_start, seg_end = seg_map[word]
            # Check if there's a competing word from sliding window
            if slide_only:
                max_slide_conf = max(slide_map.get(w, (0, 0, 0))[0] for w in slide_only)
                if seg_conf > max_slide_conf + self.confidence_threshold:
                    self.stats['segmenter_wins'] += 1
                    combined.append((word, seg_conf, seg_start, seg_end))
                else:
                    self.stats['disagreements'] += 1
            else:
                self.stats['segmenter_wins'] += 1
                combined.append((word, seg_conf, seg_start, seg_end))

        for word in slide_only:
            slide_conf, slide_start, slide_end = slide_map[word]
            # Check if there's a competing word from segmenter
            if seg_only:
                max_seg_conf = max(seg_map.get(w, (0, 0, 0))[0] for w in seg_only)
                if slide_conf > max_seg_conf + self.confidence_threshold:
                    self.stats['sliding_wins'] += 1
                    combined.append((word, slide_conf, slide_start, slide_end))
                else:
                    self.stats['disagreements'] += 1
            else:
                self.stats['sliding_wins'] += 1
                combined.append((word, slide_conf, slide_start, slide_end))

        self.stats['total_detections'] += len(combined)

        # Sort by start frame
        combined.sort(key=lambda x: x[2])
        return combined

    def _combine_segmenter_primary(
        self,
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Prefer segmenter results unless confidence is too low,
        then fall back to sliding window.
        """
        combined = []

        for word, conf, start, end in segmenter_results:
            if conf >= settings.SLIDING_WINDOW_MIN_CONFIDENCE:
                self.stats['segmenter_wins'] += 1
                combined.append((word, conf, start, end))
            else:
                # Segmenter confidence too low, check sliding window
                slide_detection = next(
                    ((w, c, s, e) for w, c, s, e in sliding_results if w == word),
                    None
                )
                if slide_detection:
                    self.stats['agreements'] += 1
                    # Use sliding window confidence with small boost
                    boosted_conf = min(1.0, slide_detection[1] + 0.05)
                    combined.append((word, boosted_conf, slide_detection[2], slide_detection[3]))

        # Add sliding window detections not in segmenter results
        seg_words = {word for word, _, _, _ in segmenter_results}
        for word, conf, start, end in sliding_results:
            if word not in seg_words:
                self.stats['sliding_wins'] += 1
                combined.append((word, conf, start, end))

        self.stats['total_detections'] += len(combined)
        return combined

    def _combine_sliding_primary(
        self,
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Prefer sliding window results unless confidence is too low,
        then fall back to segmenter.
        """
        combined = []

        for word, conf, start, end in sliding_results:
            if conf >= settings.SLIDING_WINDOW_MIN_CONFIDENCE:
                self.stats['sliding_wins'] += 1
                combined.append((word, conf, start, end))
            else:
                # Sliding confidence too low, check segmenter
                seg_detection = next(
                    ((w, c, s, e) for w, c, s, e in segmenter_results if w == word),
                    None
                )
                if seg_detection:
                    self.stats['agreements'] += 1
                    # Use segmenter confidence with small boost
                    boosted_conf = min(1.0, seg_detection[1] + 0.05)
                    combined.append((word, boosted_conf, seg_detection[2], seg_detection[3]))

        # Add segmenter detections not in sliding window results
        slide_words = {word for word, _, _, _ in sliding_results}
        for word, conf, start, end in segmenter_results:
            if word not in slide_words:
                self.stats['segmenter_wins'] += 1
                combined.append((word, conf, start, end))

        self.stats['total_detections'] += len(combined)
        return combined

    def _combine_adaptive(
        self,
        segmenter_results: List[Tuple[str, float, int, int]],
        sliding_results: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, int, int]]:
        """
        Adaptive strategy - intelligently combines detections based on:
        - Temporal overlap (IoU)
        - Agreement (same word)
        - Confidence levels
        - Context awareness

        Strategy:
        1. High IoU + Same word → BOOST confidence (strong agreement)
        2. Low IoU + Same word → Keep both (likely 2 occurrences)
        3. High IoU + Different words → Take higher confidence (conflict resolution)
        4. Only one detector → Take if confidence > threshold
        """
        combined = []
        used_seg_indices = set()
        used_slide_indices = set()

        # Match detections based on temporal overlap
        for seg_idx, (seg_word, seg_conf, seg_start, seg_end) in enumerate(segmenter_results):
            best_match = None
            best_iou = 0.0
            best_slide_idx = -1

            # Find sliding detections with temporal overlap
            for slide_idx, (slide_word, slide_conf, slide_start, slide_end) in enumerate(sliding_results):
                if slide_idx in used_slide_indices:
                    continue

                # Calculate temporal IoU
                iou = self._calculate_temporal_iou(seg_start, seg_end, slide_start, slide_end)

                if iou > best_iou:
                    best_iou = iou
                    best_match = (slide_word, slide_conf, slide_start, slide_end)
                    best_slide_idx = slide_idx

            # Decision based on IoU and word agreement
            if best_match and best_iou > settings.HYBRID_OVERLAP_THRESHOLD:
                slide_word, slide_conf, slide_start, slide_end = best_match

                if seg_word == slide_word:
                    # STRONG AGREEMENT: Same word + high IoU
                    self.stats['agreements'] += 1
                    self.stats['total_boosted'] += 1

                    # Log match
                    logger.log_hybrid_match(seg_word, seg_conf, (seg_start, seg_end),
                                          slide_word, slide_conf, (slide_start, slide_end),
                                          best_iou, agreement=True)

                    # Calibrate confidence based on IoU strength
                    max_conf = max(seg_conf, slide_conf)
                    if best_iou > 0.7:
                        # Very high overlap
                        boosted_conf = min(1.0, max_conf + self.agreement_boost)
                    else:
                        # Moderate overlap
                        boosted_conf = min(1.0, max_conf + self.agreement_boost * 0.7)

                    # Use boundaries from higher-confidence detection
                    if seg_conf > slide_conf:
                        combined.append((seg_word, boosted_conf, seg_start, seg_end))
                        self.stats['segmenter_wins'] += 1
                        logger.log_hybrid_decision(seg_word, boosted_conf,
                                                  f"Segmenter boundaries (higher conf)", boosted=True)
                    else:
                        combined.append((slide_word, boosted_conf, slide_start, slide_end))
                        self.stats['sliding_wins'] += 1
                        logger.log_hybrid_decision(slide_word, boosted_conf,
                                                  f"Sliding boundaries (higher conf)", boosted=True)

                    used_seg_indices.add(seg_idx)
                    used_slide_indices.add(best_slide_idx)

                else:
                    # CONFLICT: Different words, high IoU
                    self.stats['conflicts'] += 1

                    # Log conflict
                    logger.log_hybrid_match(seg_word, seg_conf, (seg_start, seg_end),
                                          slide_word, slide_conf, (slide_start, slide_end),
                                          best_iou, agreement=False)

                    # Take higher confidence, but apply penalty for disagreement
                    if seg_conf > slide_conf + 0.1:  # Segmenter significantly more confident
                        combined.append((seg_word, seg_conf * 0.95, seg_start, seg_end))
                        self.stats['segmenter_wins'] += 1
                        logger.log_hybrid_decision(seg_word, seg_conf * 0.95,
                                                  f"Segmenter more confident ({seg_conf:.2%} vs {slide_conf:.2%})")
                        used_seg_indices.add(seg_idx)
                    elif slide_conf > seg_conf + 0.1:  # Sliding significantly more confident
                        combined.append((slide_word, slide_conf * 0.95, slide_start, slide_end))
                        self.stats['sliding_wins'] += 1
                        logger.log_hybrid_decision(slide_word, slide_conf * 0.95,
                                                  f"Sliding more confident ({slide_conf:.2%} vs {seg_conf:.2%})")
                        used_slide_indices.add(best_slide_idx)
                    else:
                        # Similar confidence - keep both (might be overlapping gestures)
                        combined.append((seg_word, seg_conf * 0.9, seg_start, seg_end))
                        combined.append((slide_word, slide_conf * 0.9, slide_start, slide_end))
                        self.stats['disagreements'] += 1
                        logger.log_hybrid_decision(f"{seg_word}+{slide_word}", seg_conf * 0.9,
                                                  f"Similar confidence - keeping both")
                        used_seg_indices.add(seg_idx)
                        used_slide_indices.add(best_slide_idx)

            elif best_match and seg_word == best_match[0]:
                # LOW IoU but SAME WORD: Likely 2 separate occurrences
                # Keep both if confidence is decent
                if seg_conf >= 0.6:
                    combined.append((seg_word, seg_conf, seg_start, seg_end))
                    self.stats['segmenter_wins'] += 1
                    used_seg_indices.add(seg_idx)
                if best_match[1] >= 0.6:
                    combined.append((best_match[0], best_match[1], best_match[2], best_match[3]))
                    self.stats['sliding_wins'] += 1
                    used_slide_indices.add(best_slide_idx)

            else:
                # ONLY SEGMENTER detected (no match or low IoU)
                solo_threshold = settings.MIN_CONFIDENCE_THRESHOLD * settings.HYBRID_SOLO_DETECTION_MULTIPLIER
                accepted = seg_conf >= solo_threshold

                logger.log_hybrid_solo_detection("SEGMENTER", seg_word, seg_conf,
                                                solo_threshold, accepted)

                if accepted:
                    combined.append((seg_word, seg_conf, seg_start, seg_end))
                    self.stats['segmenter_wins'] += 1
                    used_seg_indices.add(seg_idx)

        # Add sliding-only detections that weren't matched
        for slide_idx, (slide_word, slide_conf, slide_start, slide_end) in enumerate(sliding_results):
            if slide_idx not in used_slide_indices:
                solo_threshold = settings.MIN_CONFIDENCE_THRESHOLD * settings.HYBRID_SOLO_DETECTION_MULTIPLIER
                accepted = slide_conf >= solo_threshold

                logger.log_hybrid_solo_detection("SLIDING", slide_word, slide_conf,
                                                solo_threshold, accepted)

                if accepted:
                    combined.append((slide_word, slide_conf, slide_start, slide_end))
                    self.stats['sliding_wins'] += 1

        self.stats['total_detections'] += len(combined)

        # Sort by start frame
        combined.sort(key=lambda x: x[2])

        # Temporal deduplication
        combined = self._deduplicate_temporal(combined)

        # Log final stats
        logger.log_hybrid_stats(self.stats)

        return combined

    def _calculate_temporal_iou(
        self,
        start1: int,
        end1: int,
        start2: int,
        end2: int
    ) -> float:
        """
        Calculate Intersection over Union for temporal segments.

        IoU = |intersection| / |union|

        Args:
            start1, end1: First temporal segment (frames)
            start2, end2: Second temporal segment (frames)

        Returns:
            IoU value between 0.0 and 1.0

        Example:
            seg1 = [10, 30]  # 20 frames
            seg2 = [20, 40]  # 20 frames
            intersection = [20, 30]  # 10 frames
            union = [10, 40]  # 30 frames
            IoU = 10 / 30 = 0.33
        """
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection_length = max(0, intersection_end - intersection_start)

        # Calculate union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_length = union_end - union_start

        # Avoid division by zero
        if union_length == 0:
            return 0.0

        # IoU
        iou = intersection_length / union_length
        return iou


    def _deduplicate_temporal(
        self,
        detections: List[Tuple[str, float, int, int]]
    ) -> List[Tuple[str, float, int, int]]:
        """
        Remove duplicate detections based on:
        - Same word
        - Temporal proximity (< 30 frames / ~1 second)
        - Keep higher confidence
        """
        if len(detections) <= 1:
            return detections

        deduplicated = []
        skip_indices = set()

        for i, (word1, conf1, start1, end1) in enumerate(detections):
            if i in skip_indices:
                continue

            # Check for duplicates ahead
            is_duplicate = False
            for j in range(i + 1, len(detections)):
                if j in skip_indices:
                    continue

                word2, conf2, start2, end2 = detections[j]

                # Check if same word and temporally close
                if word1 == word2:
                    temporal_distance = abs(start2 - end1)

                    if temporal_distance < 30:  # Less than 1 second apart
                        # It's a duplicate - keep higher confidence
                        if conf2 > conf1:
                            is_duplicate = True
                            skip_indices.add(i)
                        else:
                            skip_indices.add(j)
                        break

            if not is_duplicate:
                deduplicated.append((word1, conf1, start1, end1))

        return deduplicated

    def _deduplicate_by_word_with_temporal(
        self,
        detections: List[Tuple[str, float, int, int]],
        strategy: str = None
    ) -> List[Tuple[str, float, int, int]]:
        """
        Remove duplicate detections of same word with temporal overlap.

        Only removes duplicates if:
        1. Same word
        2. Temporal overlap (IoU > threshold)

        This allows same word to appear twice if they're temporally separated.

        Args:
            detections: List of (word, confidence, start_frame, end_frame) tuples
            strategy: Deduplication strategy ('max_confidence', 'first', 'merge')

        Returns:
            Deduplicated list preserving temporal order
        """
        if not settings.HYBRID_WORD_DEDUP_ENABLED:
            return detections

        if len(detections) <= 1:
            return detections

        strategy = strategy or settings.HYBRID_DEDUP_STRATEGY
        deduplicated = []

        for i, (word, conf, start, end) in enumerate(detections):
            # Check if this detection overlaps with any already in deduplicated
            overlaps_existing = False

            for j, (prev_word, prev_conf, prev_start, prev_end) in enumerate(deduplicated):
                if word == prev_word:
                    # Same word - check temporal overlap
                    iou = self._calculate_temporal_iou(start, end, prev_start, prev_end)

                    if iou > settings.HYBRID_OVERLAP_THRESHOLD:
                        # Overlapping duplicate - handle according to strategy
                        overlaps_existing = True

                        if strategy == 'max_confidence':
                            if conf > prev_conf:
                                # Replace with higher confidence detection
                                deduplicated[j] = (word, conf, start, end)

                        elif strategy == 'merge':
                            # Average confidence, merge temporal boundaries
                            merged_conf = (conf + prev_conf) / 2.0
                            merged_start = min(start, prev_start)
                            merged_end = max(end, prev_end)
                            deduplicated[j] = (word, merged_conf, merged_start, merged_end)

                        # 'first' strategy: do nothing, keep existing
                        break

            if not overlaps_existing:
                # No temporal overlap with existing - add as new detection
                deduplicated.append((word, conf, start, end))

        # Re-sort after deduplication
        deduplicated.sort(key=lambda x: x[2])
        return deduplicated

    def _deduplicate_by_word(
        self,
        detections: List[Tuple[str, float]],
        strategy: str = None
    ) -> List[Tuple[str, float]]:
        """
        Remove duplicate detections of same word.

        DEPRECATED: Use _deduplicate_by_word_with_temporal for temporal-aware deduplication.

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

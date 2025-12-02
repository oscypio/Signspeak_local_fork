# -*- coding: utf-8 -*-
"""
Unit tests for WordSegmenterV2 (hysteresis + adaptive thresholds + tail refinement).
"""

import numpy as np

import pytest

from ...segmentation.WordSegmenterV2 import WordSegmenterV2


def make_step_sequence(T_move=20, T_silence=8, F=4, move_amp=0.02):
    """Create a synthetic sequence: steady movement then silence."""
    seq = []
    # movement: increasing small offsets
    base = np.zeros(F, dtype=np.float32)
    for t in range(T_move):
        base = base + (move_amp / F)
        seq.append(base.copy())
    # silence: keep constant
    for _ in range(T_silence):
        seq.append(base.copy())
    return np.stack(seq, axis=0)


@pytest.mark.unit
def test_segmenter_v2_basic_cut():
    """Ensure segment is produced for sufficient movement followed by silence.

    We lower low_floor and high_multiplier to allow the small synthetic movement to cross thresholds.
    Increase move_amp so movement norm > high threshold early.
    """
    seg = WordSegmenterV2(
        silence_frames=6,
        min_word_frames=8,
        hist_len=20,
        low_pctl=0.5,
        tail_margin=0.0,
        low_floor=0.001,        # allow adaptive low to drop below original 0.12
        high_multiplier=1.2,    # easier to cross HIGH
    )
    # Increase amplitude & feature dimension for clearer motion
    seq = make_step_sequence(T_move=20, T_silence=10, F=8, move_amp=0.2)

    out = None
    for t in range(seq.shape[0]):
        out = seg.add_frame(seq[t])
        if out is not None:
            break

    assert out is not None, "Expected a segment to be returned after sufficient silence"
    assert out.shape[0] >= 8


@pytest.mark.unit
def test_segmenter_v2_too_short_ignored():
    """Short movement (< min_word_frames) should not yield a segment even if thresholds are crossed."""
    seg = WordSegmenterV2(
        silence_frames=6,
        min_word_frames=12,
        hist_len=20,
        low_floor=0.001,
        high_multiplier=1.2,
    )
    # short movement then silence (only 6 movement frames < min_word_frames=12)
    seq = make_step_sequence(T_move=6, T_silence=10, F=8, move_amp=0.2)

    out = None
    for t in range(seq.shape[0]):
        out = seg.add_frame(seq[t])
        if out is not None:
            break

    assert out is None, "Too-short segments should not be returned"

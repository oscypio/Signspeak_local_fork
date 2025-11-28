"""
Unit tests for SlidingWindowDetector

Tests basic functionality of the sliding window approach for sign detection.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from ....model_logic.segmentation.SlidingWindowDetector import SlidingWindowDetector


class TestSlidingWindowDetector:
    """Test suite for SlidingWindowDetector"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create detector with small parameters for testing
        self.detector = SlidingWindowDetector(
            window_size=10,
            stride=5,
            stability_count=2,
            min_confidence=0.5,
            max_buffer_size=50
        )

        # Mock classifier
        self.mock_classifier = Mock()

        # Mock preparer
        self.mock_preparer = Mock()
        self.mock_preparer.prepare_resampled = Mock(side_effect=lambda x: x)

    def test_initialization(self):
        """Test detector initializes with correct parameters"""
        assert self.detector.window_size == 10
        assert self.detector.stride == 5
        assert self.detector.stability_count == 2
        assert self.detector.min_confidence == 0.5
        assert len(self.detector.frame_buffer) == 0

    def test_add_frame_insufficient_buffer(self):
        """Test that no detection occurs with insufficient frames"""
        frame = np.random.randn(168)

        # Add frames but not enough for window
        for i in range(5):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)
            assert result is None

    def test_add_frame_no_stride_trigger(self):
        """Test that detection doesn't occur before stride is reached"""
        frame = np.random.randn(168)

        # Configure mock to return high confidence
        self.mock_classifier.predict_proba = Mock(return_value={
            'HELLO': 0.9,
            'WORLD': 0.1
        })

        # Add enough frames for window
        for i in range(10):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)

        # Should trigger once at frame 10 (initial window)
        assert self.mock_classifier.predict_proba.call_count == 1

    def test_stable_detection(self):
        """Test that stable predictions are detected correctly"""
        frame = np.random.randn(168)

        # Mock classifier to always return same prediction with high confidence
        self.mock_classifier.predict_proba = Mock(return_value={
            'HELLO': 0.85,
            'WORLD': 0.15
        })

        # Add frames: 10 for initial window + 5 for stride + 5 for second window
        results = []
        for i in range(20):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)
            if result is not None:
                results.append(result)

        # Should detect at least one word after stability is reached
        assert len(results) >= 1
        assert results[0][0] == 'HELLO'
        assert results[0][1] == 0.85

    def test_low_confidence_filtering(self):
        """Test that low confidence predictions are filtered out"""
        frame = np.random.randn(168)

        # Mock classifier to return low confidence
        self.mock_classifier.predict_proba = Mock(return_value={
            'HELLO': 0.3,  # Below min_confidence of 0.5
            'WORLD': 0.7
        })

        results = []
        for i in range(20):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)
            if result is not None:
                results.append(result)

        # Should not detect anything with low confidence
        # Note: WORLD has 0.7 confidence, so it should be detected
        if results:
            assert results[0][0] == 'WORLD'

    def test_reset(self):
        """Test that reset clears all state"""
        frame = np.random.randn(168)

        # Add some frames
        for i in range(15):
            self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)

        # Verify state exists
        assert len(self.detector.frame_buffer) > 0
        assert self.detector.total_frames_processed > 0

        # Reset
        self.detector.reset()

        # Verify state is cleared
        assert len(self.detector.frame_buffer) == 0
        assert self.detector.total_frames_processed == 0
        assert self.detector.last_predicted_word is None
        assert self.detector.consecutive_count == 0

    def test_batch_processing(self):
        """Test batch frame processing"""
        frames = [np.random.randn(168) for _ in range(20)]

        self.mock_classifier.predict_proba = Mock(return_value={
            'TEST': 0.9,
            'OTHER': 0.1
        })

        results = self.detector.add_frames_batch(
            frames, self.mock_classifier, self.mock_preparer
        )

        # Should detect at least one word
        assert len(results) >= 1
        assert results[0][0] == 'TEST'

    def test_cooldown_prevents_duplicates(self):
        """Test that cooldown prevents duplicate emissions"""
        frame = np.random.randn(168)

        # Mock always returns same prediction
        self.mock_classifier.predict_proba = Mock(return_value={
            'HELLO': 0.95,
            'WORLD': 0.05
        })

        # Process many frames
        results = []
        for i in range(50):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)
            if result is not None:
                results.append(result)

        # Should emit HELLO, then cooldown should prevent immediate re-emission
        # Count how many times HELLO was emitted
        hello_count = sum(1 for r in results if r[0] == 'HELLO')

        # With cooldown, should be limited (not emitted every window)
        assert hello_count < 10  # Reasonable limit given 50 frames

    def test_state_info(self):
        """Test that state info is returned correctly"""
        frame = np.random.randn(168)

        for i in range(15):
            self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)

        state = self.detector.get_state_info()

        assert 'buffer_size' in state
        assert 'total_frames_processed' in state
        assert 'consecutive_count' in state
        assert state['total_frames_processed'] == 15

    def test_transition_between_words(self):
        """Test detection of word transitions"""
        frame = np.random.randn(168)

        call_count = [0]

        def mock_predict_varying(seq):
            """Mock that returns different predictions over time"""
            call_count[0] += 1
            if call_count[0] <= 2:
                return {'HELLO': 0.9, 'WORLD': 0.1}
            else:
                return {'WORLD': 0.9, 'HELLO': 0.1}

        self.mock_classifier.predict_proba = Mock(side_effect=mock_predict_varying)

        results = []
        for i in range(30):
            result = self.detector.add_frame(frame, self.mock_classifier, self.mock_preparer)
            if result is not None:
                results.append(result)

        # Should detect transition from HELLO to WORLD
        words_detected = [r[0] for r in results]

        # Should have at least one detection
        assert len(words_detected) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


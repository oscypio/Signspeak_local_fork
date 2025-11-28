# Sliding Window Detector - Usage Guide

## Overview

The Sliding Window Detector is an alternative approach to ASL sign recognition that doesn't rely on motion-based word boundary detection. Instead of using a segmenter to find where words start and end, it classifies overlapping temporal windows and detects words through prediction stability.

## Key Advantages

✅ **Independent of segmentation accuracy** - Doesn't need precise word boundaries
✅ **Natural false positive filtering** - Requires stable predictions across multiple windows
✅ **Works with existing classifier** - No additional training needed
✅ **Configurable via environment variables** - Easy to toggle and tune

## How It Works

1. **Buffer frames** as they arrive from the frontend
2. **Extract windows** of fixed size (e.g., 60 frames) with configurable stride (e.g., every 10 frames)
3. **Classify each window** independently using your existing GRU model
4. **Track stability** - count consecutive windows with the same prediction
5. **Emit word** when stability threshold is reached (e.g., 3 consecutive windows)
6. **Apply cooldown** to prevent duplicate emissions

### Example Timeline

```
Frames: [----HELLO-----][pause][---NEED---][pause]

Window 1 (frames 0-60):   HELLO (conf: 0.82)
Window 2 (frames 10-70):  HELLO (conf: 0.89)  
Window 3 (frames 20-80):  HELLO (conf: 0.91) ← 3 consecutive! Emit "HELLO"
Window 4 (frames 30-90):  NOISE (conf: 0.45) ← Below threshold, ignored
Window 5 (frames 40-100): NEED  (conf: 0.88)
Window 6 (frames 50-110): NEED  (conf: 0.92)
Window 7 (frames 60-120): NEED  (conf: 0.89) ← 3 consecutive! Emit "NEED"
```

## Configuration

All settings are controlled via environment variables:

### Basic Configuration

```bash
# Enable sliding window mode (default: False)
USE_SLIDING_WINDOW=True

# Window size in frames - should match your model's expected input (default: 60)
SLIDING_WINDOW_SIZE=60

# Stride between windows - lower = more overlap, higher accuracy (default: 10)
SLIDING_WINDOW_STRIDE=10

# Consecutive windows needed for stable detection (default: 3)
SLIDING_WINDOW_STABILITY_COUNT=3

# Minimum confidence to accept a prediction (default: 0.5)
SLIDING_WINDOW_MIN_CONFIDENCE=0.5
```

### Advanced Configuration

```bash
# Maximum frames to buffer (prevents memory issues) (default: 300)
SLIDING_WINDOW_MAX_BUFFER=300

# Use batch prediction for better performance (default: True)
SLIDING_WINDOW_BATCH_PREDICT=True
```

## Usage Examples

### Example 1: Docker Compose

Update your `docker-compose.yml` or `.env` file:

```yaml
environment:
  - USE_SLIDING_WINDOW=True
  - SLIDING_WINDOW_SIZE=60
  - SLIDING_WINDOW_STRIDE=10
  - SLIDING_WINDOW_STABILITY_COUNT=3
  - SLIDING_WINDOW_MIN_CONFIDENCE=0.6
```

### Example 2: Direct Python

```python
import os

# Set before importing your app
os.environ['USE_SLIDING_WINDOW'] = 'True'
os.environ['SLIDING_WINDOW_STRIDE'] = '8'  # More aggressive overlap
os.environ['SLIDING_WINDOW_MIN_CONFIDENCE'] = '0.7'  # Stricter filtering

from app.model_logic.PipelineManager import PipelineManager

pipeline = PipelineManager()
# Now uses sliding window automatically
responses = pipeline.process(frames)
```

### Example 3: Testing Both Approaches

```python
# Test traditional segmenter
os.environ['USE_SLIDING_WINDOW'] = 'False'
os.environ['USE_SEGMENTER'] = 'True'
pipeline_segmenter = PipelineManager()
results_segmenter = pipeline_segmenter.process(frames)

# Test sliding window
os.environ['USE_SLIDING_WINDOW'] = 'True'
pipeline_sliding = PipelineManager()
results_sliding = pipeline_sliding.process(frames)

# Compare
print(f"Segmenter detected: {results_segmenter}")
print(f"Sliding detected: {results_sliding}")
```

## Parameter Tuning Guide

### For Higher Accuracy (slower, more conservative)

```bash
SLIDING_WINDOW_STRIDE=5           # More overlap
SLIDING_WINDOW_STABILITY_COUNT=4  # Require more consecutive windows
SLIDING_WINDOW_MIN_CONFIDENCE=0.7 # Higher confidence threshold
```

### For Lower Latency (faster, more aggressive)

```bash
SLIDING_WINDOW_STRIDE=15           # Less overlap
SLIDING_WINDOW_STABILITY_COUNT=2   # Fewer windows needed
SLIDING_WINDOW_MIN_CONFIDENCE=0.4  # Lower confidence threshold
```

### For Real-time Performance

```bash
SLIDING_WINDOW_BATCH_PREDICT=True  # Always use batch mode
SLIDING_WINDOW_MAX_BUFFER=200      # Limit buffer size
```

## Performance Considerations

### Computational Cost

- **Stride = 10**: Classifier runs every 10 frames (~3x per second at 30fps)
- **Stride = 5**: Classifier runs every 5 frames (~6x per second at 30fps)
- **Batch prediction** (recommended): Processes multiple windows at once, ~2-3x faster

### Memory Usage

- Buffer size = `SLIDING_WINDOW_MAX_BUFFER * feature_dim`
- For 300 frames × 168 features × 4 bytes = ~200KB (negligible)

### Latency

Minimum latency = `window_size + (stability_count - 1) * stride`

Examples:
- Default (60 + 2×10 = 80 frames) ≈ 2.7 seconds at 30fps
- Fast mode (60 + 1×15 = 75 frames) ≈ 2.5 seconds at 30fps
- Accurate mode (60 + 3×5 = 75 frames) ≈ 2.5 seconds at 30fps

## Switching Between Modes

You can switch between traditional segmenter and sliding window at runtime:

```python
from app.model_logic.utils.config import settings

# Switch to sliding window
settings.USE_SLIDING_WINDOW = True

# Switch back to traditional segmenter
settings.USE_SLIDING_WINDOW = False
settings.USE_SEGMENTATOR = True
```

## Debugging

Get detector state info:

```python
pipeline = PipelineManager()
# ... process some frames ...

if settings.USE_SLIDING_WINDOW:
    state = pipeline.sliding_detector.get_state_info()
    print(f"Buffer size: {state['buffer_size']}")
    print(f"Last prediction: {state['last_predicted_word']} ({state['last_confidence']:.2f})")
    print(f"Consecutive count: {state['consecutive_count']}")
    print(f"Total frames: {state['total_frames_processed']}")
```

## Comparing Accuracy

To evaluate which approach works better for your data:

1. Set up evaluation script:
```python
import os
from eval.pkl_batch_evaluator import evaluate_pkl_batch

# Test segmenter
os.environ['USE_SLIDING_WINDOW'] = 'False'
accuracy_segmenter = evaluate_pkl_batch(test_files)

# Test sliding window
os.environ['USE_SLIDING_WINDOW'] = 'True'
accuracy_sliding = evaluate_pkl_batch(test_files)

print(f"Segmenter accuracy: {accuracy_segmenter:.2%}")
print(f"Sliding window accuracy: {accuracy_sliding:.2%}")
```

## Common Issues & Solutions

### Issue: Too many duplicate detections
**Solution**: Increase `SLIDING_WINDOW_STABILITY_COUNT` or `SLIDING_WINDOW_MIN_CONFIDENCE`

### Issue: Missing short words
**Solution**: Decrease `SLIDING_WINDOW_SIZE` (e.g., to 40-50 frames)

### Issue: High latency
**Solution**: Increase `SLIDING_WINDOW_STRIDE` or decrease `SLIDING_WINDOW_STABILITY_COUNT`

### Issue: False positives
**Solution**: Increase `SLIDING_WINDOW_MIN_CONFIDENCE` to 0.7-0.8

### Issue: Memory usage too high
**Solution**: Decrease `SLIDING_WINDOW_MAX_BUFFER`

## Integration with Existing Code

The sliding window detector integrates seamlessly with your existing pipeline:

- ✅ Works with `force_end_sentence()` for PUSH/end-of-sentence handling
- ✅ Works with `reset_buffer()` for buffer management
- ✅ Uses same `ASLClassifier` and `UnifiedDataPreparer`
- ✅ Returns same response format as traditional segmenter
- ✅ Compatible with polishing models (T5/Qwen)

## Recommended Starting Configuration

```bash
# Good balance of accuracy and performance
USE_SLIDING_WINDOW=True
SLIDING_WINDOW_SIZE=60
SLIDING_WINDOW_STRIDE=10
SLIDING_WINDOW_STABILITY_COUNT=3
SLIDING_WINDOW_MIN_CONFIDENCE=0.6
SLIDING_WINDOW_BATCH_PREDICT=True
```

Start with this and tune based on your specific data and requirements.

## Next Steps

1. Test on your existing evaluation dataset
2. Compare accuracy with traditional segmenter
3. Tune parameters based on results
4. Consider hybrid approach (use both and select best confidence)

For questions or issues, check the code documentation in `SlidingWindowDetector.py`.


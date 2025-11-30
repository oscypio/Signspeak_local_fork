# SignSpeak Real-time Detection System

Real-time webcam-based sign language detection using ML API.

## Requirements

```bash
pip install opencv-python mediapipe numpy
```

**Note:** You don't need torch locally - the ML API handles all model inference.

## Usage

**1. Start the ML API server:**

```bash
# From SignSpeak-ML directory
uvicorn app.main:app --reload
```

The API will start on `http://localhost:8000`

**2. Run the realtime detection:**

```bash
# From SignSpeak-ML directory (in a new terminal)
python eval/realtime_test.py
```

## Configuration

Edit `eval/realtime_test.py` in the `Config` class:

- **BATCH_SIZE**: Frames to buffer before processing (default: 10, optimized for stride=10)
- **CAMERA_INDEX**: Camera device index (default: 0)
- **SHOW_RAW_PREDICTIONS**: Show scanning/stability info (default: True)

## Controls

- **Q**: Quit
- **SPACE**: Manual sentence end (same as PUSH sign)
- **R**: Reset current sentence

## Display

### Top Panel
- **Building**: Words being accumulated
- **Sentence**: Final polished sentence
- **Stats**: Words/Sentences/Time

### Bottom Panel
- **► Scanning**: Current raw prediction from detector
- **► Stability**: Consecutive windows with same prediction (X/3)
- **► Detector buffer**: Frames in detector's internal buffer
- **FPS**: Current frame rate
- **Batch**: Progress bar showing buffer fill (0/10 → 10/10)

## Architecture

```
Camera → MediaPipe → FrameData (JSON)
                        ↓
                   [Buffer: 10 frames]
                        ↓
              HTTP POST /api/predict_landmarks
                        ↓
                 PipelineManager (server-side)
                        ↓
              SlidingWindowDetector
               (stride=10, window=70)
                        ↓
                  ASLClassifier
                        ↓
              SentencePolisher (LLM)
                        ↓
              HTTP Response (JSON)
                        ↓
                   Final output
```

**Benefits of API mode:**
- No need for local model files
- No GPU/torch required on client
- Matches production architecture (backend → ML API)
- Easy to test latency and network overhead
- **Continuous frame stream** - sends frames even when no hands detected (empty landmarks), matching real app behavior

## Settings from .env

The script automatically uses settings from `.env`:

- `USE_SLIDING_WINDOW = True` (current mode)
- `SLIDING_WINDOW_STRIDE = 10` (default)
- `SLIDING_WINDOW_STABILITY_COUNT = 3` (consecutive windows required)
- `SLIDING_WINDOW_MIN_CONFIDENCE = 0.7` (threshold)
- `SPECIAL_LABEL = 'PUSH'` (ends sentence)

## Troubleshooting

### No camera found
- Check `CAMERA_INDEX` in Config (try 0 or 1)
- Ensure camera is not used by another app

### Low FPS
- Reduce `BATCH_SIZE` (but not below stride value)
- Disable `SHOW_LANDMARKS` in Config
- Check GPU availability (model uses `settings.DEVICE`)

### No words detected
- Check lighting (hands must be clearly visible)
- Ensure MediaPipe confidence is appropriate (0.7)
- Verify model path in `.env`: `MODEL_PATH`

### ImportError
- Run from `SignSpeak-ML/` directory, not `eval/`
- Check all dependencies are installed

## Output Example

```
==============================================================
SignSpeak Real-time Detection System
==============================================================

[1/3] Loading ML Pipeline...
  ✓ Pipeline loaded
  ✓ Mode: SlidingWindow
  ✓ Window: 70 frames, stride: 10
  ✓ Stability: 3 consecutive windows required
  ✓ Classes: 150 words

[2/3] Initializing MediaPipe Hands...
  ✓ MediaPipe ready (max 2 hands, confidence 0.7)

[3/3] Opening camera...
  ✓ Camera opened (640x480 @ 30fps)

==============================================================
✓ System Ready!
==============================================================

  ➤ Word: HELLO (conf: 0.85)
  ➤ Word: MY (conf: 0.78)
  ➤ Word: NAME (conf: 0.82)

  ✓ Sentence: Hello, my name is...

Session Statistics:
  Duration: 01:23.12
  Total Frames: 2494
  Words Detected: 18
  Sentences: 3
  Avg FPS: 30.1

✓ Cleanup complete. Goodbye!
```

## Continuous Frame Stream

**Important:** This script sends frames **continuously**, even when no hands are detected:

✅ **With hands:** `landmarks: [[{x, y, z, ...}, ...]], handedness: [[{...}]]`  
✅ **Without hands:** `landmarks: [], handedness: []`

**Why is this important?**
1. **Matches production behavior** - backend will send frames at fixed intervals (e.g., every 33ms)
2. **Temporal continuity** - segmenter needs continuous timeline to detect word boundaries
3. **Silence detection** - empty frames = "silence" between words (crucial for segmentation!)
4. **No overlap needed** - pipeline maintains state between batches

This is **different** from only sending frames when hands are visible!

## Performance Tips

1. **Batch Size**: Keep at 10 (matches default stride)
2. **Camera Resolution**: 640x480 is optimal (balance quality/speed)
3. **Lighting**: Good lighting improves MediaPipe detection
4. **Hand Position**: Keep hands in frame, avoid fast movements
5. **Sign Clearly**: Pause slightly between words for stability

## Notes

- The script uses **local PipelineManager** (no API calls)
- **Raw predictions** show detector's internal state before confirmation
- **Stability count** prevents false positives (requires 3 consecutive matches)
- Pipeline maintains **state between batches** (no overlap needed from backend)


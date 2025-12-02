# SignSpeak ML API Documentation

## Table of Contents

1. **Overview**
2. **Quick Start**
3. **API Endpoints**
4. **Configuration Guide**
5. **Model Performance**
6. **Troubleshooting**

---

# 1. Overview

SignSpeak ML is a **FastAPI-based microservice** for real-time sign language recognition. It processes hand landmark data and returns detected words or complete sentences.

Attention - this Service requires proper frame buffering from the backend to function correctly in real-time scenarios.

This Service **can also be used with video-translation** by sending pre-processed (with mediapipe) videos.

## What it does

- Receives **hand landmarks** (x, y, z coordinates for 21 points per hand)
- Detects sign language gestures using **GRU-based classifier**
- Supports multiple detection modes: **Sliding Window**, **Motion Segmenter**, or **Hybrid**
- Polishes detected words into natural English sentences using **LLM** (Qwen or T5)
- Returns results via **REST API**

## Pipeline Context

This ML service is part of a larger system:
```
Frontend (MediaPipe) → Backend (Frame Buffering) → ML API → Results
```

The ML API expects batches of landmarks (typically 30-60 frames) and returns detected words/sentences.

## Key Features

- ⚡ **Real-time detection** with buffer flush mechanism
- 🔀 **Multiple detection modes** (Sliding Window, Segmenter, Hybrid)
- 🗳️ **Voting-based stability** for continuous classification
- 🎯 **Configurable confidence thresholds**
- 📊 **Detailed logging** for debugging
- 🔧 **Production-ready** with Docker support


---

# 2. Quick Start

## Prerequisites

- Docker + Docker Compose
- Python 3.10+ (if running locally)
- AI models (see below)

## Installation

### 1. Download Models

Place models in `app_models/`:

```
app_models/
├── conf_final_v3.3(hybrid_norm).pt  ← Classifier model
└── qwen2.5-1.5b-instruct-q4_k_m.gguf ← Polishing model
```

Download polishing model: [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf)

### 2. Run with Docker

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`

### 3. Verify Installation

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

## Local Development (without Docker)

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

# 3. API Endpoints

## Main Endpoint

### `POST /api/predict_landmarks`

Processes hand landmark frames and returns detected words or sentences.

**Request Body:**
```json
[
  {
    "timestamp": 1234.5,
    "sequenceNumber": 1,
    "receivedAt": 1234.5,
    "landmarks": [
      [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99},
        ...  // 21 points per hand
      ]
    ],
    "handedness": [
      [
        {"score": 0.98, "index": 0, "categoryName": "Right", "displayName": "Right"}
      ]
    ]
  }
]
```

**Response:**
```json
{
  "results": [
    {
      "prediction": "HELLO",
      "status": "word_added",
      "current_words": ["HELLO"],
      "detail": "Word added successfully"
    }
  ]
}
```

**Note:** Response is always a list. With `USE_SEGMENTATOR=False` and `USE_SLIDING_SEGMENTSTOR=False`, list contains only 1 element per request.

---

# Example 1 — The user signs “NEED”

### ➡️ Request (simplified example)

```json
[
  {
    "timestamp": 31034.3,
    "sequenceNumber": 1,
    "receivedAt": 31034.3,
    "landmarks": [
      [{ "x": 0.24, "y": 0.94, "z": 0.0 }]
    ],
    "handedness": [
      [{ "score": 0.98, "categoryName": "Right" }]
    ]
  }
]
```

### ⬅️ Response

```json
{
  "results": [
    {
      "prediction": "NEED",
      "status": "word_added",
      "current_words": ["NEED"],
      "detail": "Word added successfully"
    }
  ]
}
```

---

# Example 2 — The user signs “PHONE”

### ⬅️ Response

```json
{
  "results": [
    {
      "prediction": "PHONE",
      "status": "word_added",
      "current_words": ["NEED", "PHONE"],
      "detail": "Word added successfully"
    }
  ]
}
```

---

# Example 3 — The user signs “PUSH”

### ⬅️ Response

```json
{
  "results": [
    {
      "prediction": "PUSH",
      "status": "end_of_sentence",
      "sentence": "I need a phone.",
      "detail": "Sentence formed"
    }
  ]
}
```

---

## Utility Endpoints

These helper endpoints let you manage the internal buffering logic without sending a special sign.

### 1. Force Sentence Finalization
```
POST /api/force_end_sentence
```
Forces the backend to treat the currently buffered words as a completed sentence (as if the classifier predicted the special label `PUSH`). Useful when:
- You lost the “PUSH” frame
- You want to flush partial input manually
- You are testing sentence polishing separately

#### Response (example when buffer holds `["NEED", "PHONE"]`)
```json
{
  "results": [
    {
      "prediction": "PUSH",
      "status": "end_of_sentence",
      "sentence": "I need a phone.",
      "detail": "Sentence formed"
    }
  ]
}
```
If the word buffer is empty, it will return an empty (possibly polished) sentence string.

### 2. Reset Word Buffer
```
POST /api/reset_buffer
```
Clears the current `word_buffer` (discarding unsent words) without generating a sentence. Does NOT return an entry in `results`—only status info.

#### Response
```json
{
  "status": "ok",
  "detail": "Buffer reset"
}
```
Use this if a sequence was mis-segmented or you want to start a new sentence from scratch.

---
# Full User Journey

| User signs | Data sent              | Backend response                     | UI should show             |
|------------|------------------------|--------------------------------------|----------------------------|
| NEED       | frames of “HELLO”      | `["HELLO"]`                          | HELLO                      |
| GOVERNMENT | frames of “GOVERNMENT” | `["HELLO","GOVERNMENT"]`             | HELLO GOVERNMENT           |
| PASSPORT   | frames of “THANK YOU”  | `["HELLO","GOVERNMENT","THANK YOU"]` | HELLO GOVERNMENT THANK YOU |
| PUSH       | frames of “PUSH”       | `"Hello, thank you government."`     | Final sentence             |

---

# 4. Testing the API Yourself

## Postman / Insomnia

1. New request → POST  
2. URL: `http://localhost:8000/api/predict_landmarks`  
3. Body → raw → JSON  
4. Paste landmark data  
5. Send

---

## Python Example

```python
import requests
import json

data = json.load(open("sample.json"))
response = requests.post("http://localhost:8000/api/predict_landmarks", json=data)
print(response.json())
```

---

# 5. Where Do the AI Models Come From?

### ASL Classifier (GRU)

```
Model still under development - with basic configuration in .pt file
```

### Sentence Polisher (Qwen or T5)

```
app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```


---

# 5. Model Performance

Current model (`conf_final_v3.3(hybrid_norm).pt`) performance:

Current state (when using conf_final_v3.3(hybrid_norm).pt):

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **87.01%** |
| Avg Loss  | **0.3814** |
| Macro F1  | **0.869**  |
| Precision | **0.887**  |
| Recall    | **0.868**  |


Glossary:

| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| DRINK     | 0.857     | 0.750  | 0.800    |
| HELLO     | 1.000     | 1.000  | 1.000    |
| HELP      | 1.000     | 1.000  | 1.000    |
| HOW       | 0.833     | 1.000  | 0.909    |
| LUNCH     | 0.571     | 0.800  | 0.667    |
| MUCH      | 0.667     | 0.800  | 0.727    |
| NEED      | 1.000     | 0.750  | 0.857    |
| PHONE     | 0.800     | 0.800  | 0.800    |
| PLEASE    | 0.833     | 1.000  | 0.909    |
| PUSH      | 1.000     | 1.000  | 1.000    |
| SORRY     | 0.857     | 0.857  | 0.857    |
| THANK YOU | 1.000     | 0.800  | 0.889    |
| WAIT      | 1.000     | 1.000  | 1.000    |
| WANT      | 1.000     | 0.600  | 0.750    |


**Note:** Current Segmenter uses heuristic motion detection optimized for the training dataset. To disable automatic segmentation, set `USE_SEGMENTATOR=False` and `USE_SLIDING_WINDOW=False`, as well as `USE_HYBRID_MODE=False` - the API will then treat each request as a single word.

---

# 4. Configuration Guide

All configuration is in `/app/model_logic/utils/config.py`. Most important are also listed in `.env` file.

## Detection Mode Settings
| Variable           | Default Value | Description                                                                                           |
| ------------------ | ------------- | ----------------------------------------------------------------------------------------------------- |
| `MOTION_THRESHOLD` | `0.12`        | Minimal average motion required to consider that movement is happening. Helps detect start of a sign. |
| `SILENCE_FRAMES`   | `6`           | Number of consecutive low-motion frames needed to declare a break (silence) between signs.            |
| `MIN_WORD_FRAMES`  | `8`           | Minimum number of frames required for a valid word segment.                                           |
| `BURST_MULTIPLIER` | `2.0`         | Factor applied to motion spikes to detect rapid movements.                                            |
| `EMA_ALPHA`        | `0.40`        | Smoothing factor for the Exponential Moving Average used in motion estimation.                        |

### Data Preparer Config - /app/model_logic/utils/config.py
| Variable          | Default Value | Description                                                       |
| ----------------- | ------------- | ----------------------------------------------------------------- |
| `EXPECTED_FRAMES` | `60`          | Expected fixed length of each window/segment after preprocessing. |
| `ADD_FEATURES`    | `False`       | Whether to add extra features such as velocity/acceleration.      |

### Model and Polishing Config - /app/model_logic/utils/config.py
| Variable               | Default Value                                                               | Description                                                                    |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `DEVICE`               | `"cpu"`                                                                     | Device used for inference.                                                     |

## Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SEGMENTATOR` | `False` | Motion-based word boundary detection.<br>• `True` → auto-detect word boundaries<br>• `False` → each request = 1 word |
| `USE_SLIDING_WINDOW` | `True` | Continuous classification with voting.<br>• Recommended for real-time detection |
| `USE_HYBRID_MODE` | `False` | Combines segmenter + sliding window.<br>• Best accuracy, higher latency |
| `SPECIAL_LABEL` | `PUSH` | Word that triggers sentence finalization |
| `MIN_CONFIDENCE_THRESHOLD` | `0.6` | Master threshold for all detections<br>• 0.75 recommended for production |
| `USE_HYBRID_NORMALIZATION` | `True` | Landmark normalization method<br>• `True` → 138 features (better)<br>• `False` → 126 features<br>• **Must match training!** |

## Model Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `conf(large).pt` | GRU classifier model |
| `POLISHING_MODEL_PATH` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | LLM for grammar correction |
| `DEVICE` | `cpu` | Device for inference (`cpu` or `cuda`) |
| `USE_T5` | `False` | Use T5 instead of Qwen for polishing |

---

### 🆕 Hybrid Mode Configuration (New Feature!)

**Hybrid Mode** intelligently combines motion-based segmentation with continuous sliding window detection for optimal accuracy.

#### Enable Hybrid Mode
```env
USE_HYBRID_MODE=True
HYBRID_STRATEGY=adaptive  # Recommended
```

#### Key Hybrid Mode Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_STRATEGY` | `adaptive` | Combination strategy:<br>• `adaptive` - IoU-based matching with conflict resolution ⭐<br>• `max_confidence` - Choose highest confidence<br>• `voting` - Majority voting<br>• `segmenter_primary` - Prefer segmenter<br>• `sliding_primary` - Prefer sliding |
| `HYBRID_OVERLAP_THRESHOLD` | `0.5` | Minimum IoU (0.0-1.0) to match detections from both methods.<br>• `0.5` = 50% temporal overlap required<br>• Higher = stricter matching |
| `HYBRID_AGREEMENT_BOOST` | `0.15` | Confidence boost when both methods agree (+15%)<br>• Rewards consensus between detectors |
| `HYBRID_SOLO_DETECTION_MULTIPLIER` | `0.9` | **NEW!** Threshold multiplier for solo detections.<br>• Solo threshold = MIN_CONFIDENCE × 0.9<br>• Example: If MIN=0.75, solo needs 0.675<br>• Adjust: 0.85-0.95 |

#### How Hybrid Mode Works

1. **Both detectors run in parallel**
   - Segmenter: Motion-based detection (good for clear gestures)
   - Sliding Window: Continuous classification with voting (good for subtle gestures)

2. **Adaptive strategy matches detections**
   - High IoU + Same word → **BOOST** confidence (strong agreement)
   - High IoU + Different words → **CONFLICT** resolution (choose higher confidence)
   - Low IoU + Same word → Keep both (likely 2 separate occurrences)
   - Solo detection → Add if confidence ≥ MIN_THRESHOLD × SOLO_MULTIPLIER

3. **Temporal deduplication**
   - Removes duplicates within 30 frames (~1 second)
   - Keeps higher confidence detection

#### Example Scenarios

**Scenario 1: Agreement (Both detect same word)**
```
Segmenter: "HELLO", conf=0.85, frames=[10-35]
Sliding:   "HELLO", conf=0.88, frames=[12-34]
IoU = 0.92 (high overlap)
→ Output: "HELLO", conf=1.0 (boosted!), frames=[12-34]
```

**Scenario 2: Solo Detection (Only sliding detected)**
```
Segmenter: (nothing - motion too subtle)
Sliding:   "THANK", conf=0.72, frames=[90-110]
Check: 0.72 >= 0.75 × 0.9 = 0.675 ✅
→ Output: "THANK", conf=0.72 (sliding saved it!)
```

**Scenario 3: Conflict (Different words, same time)**
```
Segmenter: "WORLD", conf=0.75, frames=[50-75]
Sliding:   "WORK",  conf=0.78, frames=[52-74]
IoU = 0.88 (high overlap but different words)
→ Output: "WORK", conf=0.74 (higher confidence wins with penalty)
```

---

### Sliding Window Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIDING_WINDOW_STRIDE` | `5` | How often to classify (1=every frame, 5=every 5th)<br>• `1` = best accuracy, slower<br>• `5` = faster, good balance<br>• `10` = fastest, may miss quick gestures |
| `SLIDING_WINDOW_VOTING_SIZE` | `20` | Number of recent predictions for voting<br>• Larger = more stable but slower to detect |
| `SLIDING_WINDOW_VOTE_THRESHOLD` | `13` | Minimum votes to accept word (13/20 = 65%)<br>• `16` = 80% (stricter)<br>• `11` = 55% (more lenient) |
| `SLIDING_WINDOW_MIN_CONFIDENCE` | `0.55` | Confidence threshold for individual predictions<br>• Lower than MIN_CONFIDENCE_THRESHOLD |

---

### Performance Tuning Guide

#### For Higher Precision (Fewer False Positives)
```env
MIN_CONFIDENCE_THRESHOLD=0.80          # Was 0.75
SLIDING_WINDOW_VOTE_THRESHOLD=16       # Was 13 (80% consensus)
HYBRID_OVERLAP_THRESHOLD=0.6           # Was 0.5 (stricter matching)
HYBRID_SOLO_DETECTION_MULTIPLIER=0.95  # Was 0.9 (stricter solo)
```

#### For Higher Recall (Catch More Gestures)
```env
MIN_CONFIDENCE_THRESHOLD=0.65          # Was 0.75
SLIDING_WINDOW_STRIDE=1                # Was 5 (classify every frame)
SLIDING_WINDOW_VOTE_THRESHOLD=11       # Was 13 (55% consensus)
HYBRID_SOLO_DETECTION_MULTIPLIER=0.85  # Was 0.9 (more lenient solo)
```

#### For Best Performance (Speed)
```env
SLIDING_WINDOW_STRIDE=10               # Was 5 (classify every 10th frame)
SLIDING_WINDOW_VOTING_SIZE=10          # Was 20 (smaller voting window)
USE_HYBRID_MODE=False                  # Disable hybrid, use single detector
```

---

### Configuration File Structure

The configuration is now organized into **5 logical sections** in `/app/model_logic/utils/config.py`:

1. **Core Detection** - Segmenter & Sliding Window settings
2. **Hybrid Mode** - Adaptive combination parameters
3. **Model & Preprocessing** - Model paths, normalization, data preparation
4. **Performance & Buffer Management** - Flush settings, frame batching ⭐ NEW!
5. **Logging & Debugging** - Detailed logging options

Each variable includes:
- 📝 Detailed description of what it does
- 💡 Suggested values for different use cases
- ⚠️ Warnings for critical parameters (e.g., DON'T CHANGE without retraining)
- 🗑️ DEPRECATED markers for obsolete settings

---

## 🆕 Buffer Flush Mechanism (Critical for Real-Time Detection!)

### What is "Flush"?

In real-time sign language detection, frames arrive in **batches** (typically 30-45 frames every 3-5 seconds from the Java backend). The problem:

- **Segmenter** requires some frames of "silence" (no hand motion) to detect word boundaries
- **Sliding Window** builds voting consensus over time
- If a gesture is **in progress** at the end of a batch → it's incomplete

**Flush** forces the system to emit buffered detections at batch boundaries, preventing loss of the last word in a sequence. Helps motion-based segmentation to work with more stability - especially when there is no silence between signs.

### How Flush Works

When frames arrive in batches, a gesture might be incomplete at batch boundaries:

```
Batch 1: [HELLO frames... HEL]  ← incomplete at end
         ↓ Without flush: LOST (waits for silence that never comes)
         ↓ With flush: Emits "HELLO" (confidence >= FLUSH_MIN)
         
Batch 2: [LO frames... WORLD frames...]
         ↓ Continues normally
```

Flush ensures the last word in each batch is not lost while waiting for silence detection.

### Flush Configuration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_FLUSH_ON_BATCH_END` | `True` | **CRITICAL!** Enable flush at batch end.<br>• `True` → Prevents losing last word ✅<br>• `False` → Last word in batch is lost ❌<br>• **Recommendation:** Always `True` for real-time |
| `MIN_FRAMES_FOR_FLUSH` | `30` | Minimum frames in buffer before allowing flush.<br>• Too low (10-20) → May emit noise/fragments<br>• Too high (40+) → May not flush short gestures<br>• **Optimal:** 25-30 frames |
| `FLUSH_MIN_CONFIDENCE` | `0.55` | Lower confidence threshold for flushed words.<br>• Lower than `MIN_CONFIDENCE_THRESHOLD` (0.6-0.65)<br>• Catches uncertain last words that might be valid<br>• **Optimal:** 0.50-0.58 |


### Complete Flush Optimization Example

**For Real-Time Production Use:**

1. **Update `config.py`:**
```python
# Detection Mode
USE_SLIDING_WINDOW = True
USE_HYBRID_MODE = False  # Start simple
MIN_CONFIDENCE_THRESHOLD = 0.65

# Sliding Window Voting
SLIDING_WINDOW_VOTING_SIZE = 25
SLIDING_WINDOW_VOTE_THRESHOLD = 18
SLIDING_WINDOW_STRIDE = 1

# Flush Settings
FORCE_FLUSH_ON_BATCH_END = True  # CRITICAL!
MIN_FRAMES_FOR_FLUSH = 30
FLUSH_MIN_CONFIDENCE = 0.55
```



### Troubleshooting Flush Issues

| Problem | Solution                                                 |
|---------|----------------------------------------------------------|
| Last word always missing | Set `FORCE_FLUSH_ON_BATCH_END=True`                      |
| Too many false positives at batch end | Increase `FLUSH_MIN_CONFIDENCE` to 0.60+                 |
| Fragments/noise in output | Increase `MIN_FRAMES_FOR_FLUSH` to 35-40                 |
| Short gestures not detected | Decrease `MIN_FRAMES_FOR_FLUSH` to 20-25                 |
| Detections are delayed | Use faster approach - disable alternatives or change llm |



### ML Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Batch of Frames                    │
│  • Typically 30-60 frames from backend                       │
│  • Each frame: hand landmarks (42 points × 4 coords)         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              1. DATA PREPARATION (DataPreparer)              │
│  • Normalize landmarks (wrist-centered or hybrid)            │
│  • Flatten to vectors (168 or 138 features)                  │
│  • Optional: velocity + acceleration features                │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              2. DETECTION (choose mode)                      │
│                                                              │
│  [Sliding Window Mode]         [Hybrid Mode]                │
│  • Build 60-frame windows      • Run both detectors         │
│  • Classify each window        • Match by temporal IoU      │
│  • Voting: 18/20 consensus     • Boost on agreement         │
│  • Emit stable predictions     • Resolve conflicts          │
│                                                              │
│  [Segmenter Mode]                                           │
│  • Detect motion/silence                                     │
│  • Segment by boundaries                                     │
│  • Classify segments                                         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              3. FLUSH MECHANISM (batch end)                  │
│  IF FORCE_FLUSH_ON_BATCH_END = True:                         │
│    • Check detector buffers                                  │
│    • Emit pending words (confidence >= FLUSH_MIN)            │
│    • Prevents losing last word in batch                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              4. CLASSIFICATION (GRUClassifier)               │
│  • GRU-based neural network                                  │
│  • Input: (60, features) normalized sequence                 │
│  • Output: word probabilities                                │
│  • Filter by MIN_CONFIDENCE_THRESHOLD                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              5. WORD BUFFERING                               │
│  • Accumulate words: ["HELLO", "WORLD"]                      │
│  • Wait for SPECIAL_LABEL (e.g., "PUSH")                     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              6. SENTENCE POLISHING (on SPECIAL_LABEL)        │
│  • LLM (Qwen or T5) corrects grammar                         │
│  • "HELLO WORLD" → "Hello, world!"                           │
│  • Clear buffer, return sentence                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
                  RESPONSE
```

---

---

# 6. Troubleshooting

## Common Issues

| Problem | Solution |
|---------|----------|
| Last word always missing | Set `FORCE_FLUSH_ON_BATCH_END=True` |
| Too many false positives | Increase `MIN_CONFIDENCE_THRESHOLD` to 0.70+ |
| Detections too slow | Increase `SLIDING_WINDOW_STRIDE` to 5 or 10 |
| Short gestures not detected | Decrease `MIN_FRAMES_FOR_FLUSH` to 20-25 |
| Noisy/fragment detections | Increase `FLUSH_MIN_CONFIDENCE` to 0.60+ |

## Logging Configuration

```python
# For debugging
ENABLE_DETAILED_LOGGING = True
LOG_VOTING = True
LOG_FILTERING = True
LOG_HYBRID_DECISIONS = True  # If using hybrid mode

# For production
LOG_MINIMAL = True
ENABLE_DETAILED_LOGGING = False
```

## Performance Benchmarks

| Configuration | Latency | Accuracy | CPU Usage |
|--------------|---------|----------|-----------|
| Sliding Window (stride=1) | ~500ms | 87% | Medium |
| Sliding Window (stride=5) | ~300ms | 83% | Low |
| Hybrid Mode | ~700ms | 91% | High |

---

# 🚀 Quick Reference Guide

## Most Important Settings for Production - Real-Time use

### 1. Enable Flush 
```python
# config.py
FORCE_FLUSH_ON_BATCH_END = True  # Must be True for real-time
MIN_FRAMES_FOR_FLUSH = 30
FLUSH_MIN_CONFIDENCE = 0.55
```

### 2. Choose Detection Mode
```python
# Option A: Sliding Window Only (recommended for speed)
USE_SLIDING_WINDOW = True
USE_HYBRID_MODE = False

# Option B: Hybrid Mode (recommended for accuracy)
USE_HYBRID_MODE = True
```

### 3. Set Confidence Thresholds
```python
MIN_CONFIDENCE_THRESHOLD = 0.65  # Master threshold
FLUSH_MIN_CONFIDENCE = 0.55      # Lower for flush
```

## Most Important Settings for Production - Video-Translation use

### 1. Enable Flush 
```python
# config.py
FORCE_FLUSH_ON_BATCH_END = True  # preffered, but False cna also work
MIN_FRAMES_FOR_FLUSH = 30
FLUSH_MIN_CONFIDENCE = 0.4 # dependent on the video cuts
```

### 2. Choose Detection Mode
```python
# Option A: Motion-Segmenter Only (recommended for accuracy)
USE_SEGMENTER = True
USE_HYBRID_MODE = False

# Option B: Hybrid Mode (use for fast-paced videos)
USE_HYBRID_MODE = True
```

### 3. Set Confidence Thresholds
```python
MIN_CONFIDENCE_THRESHOLD = 0.75  # Master threshold
FLUSH_MIN_CONFIDENCE = 0.4      # Lower for flush (for the last word)
```
** The numbers proposed can be not optimal for your usage **

## Critical Troubleshooting Checklist

- ❌ **Last word always missing** → Set `FORCE_FLUSH_ON_BATCH_END=True`
- ❌ **Too many false positives** → Increase `MIN_CONFIDENCE_THRESHOLD` to 0.70+
- ❌ **Detections too slow** → Use faster detection method (ex. disable segmenter alternatives, etc...)
- ❌ **Short gestures not detected** → Decrease `MIN_FRAMES_FOR_FLUSH` to 20-25
- ❌ **Noisy/fragment detections** → Increase `FLUSH_MIN_CONFIDENCE` to 0.60+
- ❌ **Segmenter doesn't work** → Enable flush OR increase batch size to 60+ frames

## Logging Configuration

```python
# For debugging
ENABLE_DETAILED_LOGGING = True
LOG_VOTING = True
LOG_FILTERING = True
LOG_HYBRID_DECISIONS = True  # If using hybrid mode

# For production
LOG_MINIMAL = True
ENABLE_DETAILED_LOGGING = False
```

## Performance Benchmarks

| Configuration | Latency | Accuracy | CPU Usage |
|--------------|---------|----------|-----------|
| Sliding Window (stride=1) | ~500ms | 87% | Medium |
| Sliding Window (stride=5) | ~300ms | 83% | Low |
| Hybrid Mode | ~700ms | 91% | High |

## File Locations

- **Configuration:** `SignSpeak-ML/app/model_logic/utils/config.py`
- **Models:** `SignSpeak-ML/app_models/`
- **API:** `http://localhost:8000/api/predict_landmarks`

---

# You're Ready to Go 🎉

For questions or issues, check the logs with `LOG_MINIMAL=True` first, then enable detailed logging if needed.

**Key Takeaway:** Always keep `FORCE_FLUSH_ON_BATCH_END=True` for real-time detection!


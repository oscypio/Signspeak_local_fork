# SignSpeak API: A Simple Guide

## Table of Contents

1. **What Does This Backend Do?**  

2. **Quick Setup: Get Running in 3 Steps**  

3. **How to Use the API (The Important Part!)**  
   - The Key Endpoint: `/api/predict_landmarks`
   - Utility Endpoints: Force end & reset buffer
   - Full User Journey: A Realistic Conversation  
   
4. **Testing the API Yourself (Postman, Insomnia, or Code)**  

5. **Where Do the AI Models Come From?**

6. **Config for Development and Enhancements**
   - Model Performance Metrics
   - General Configuration
   - Architecture Flow Diagram

---

# 1. What Does This Backend Do?

Think of this backend as the **brain that understands sign language**.  
It doesn't need video — all it needs is **hand landmarks**, the (x, y, z) coordinates of 21 points per hand, frame by frame.

Here is how the system works:

1. The **MediaPipe frontend** captures hand landmarks from a webcam (~30 fps).
2. Hand landmarks are streamed via **WebSocket** to the Java backend.
3. The **Java backend** buffers frames and sends batches (45 frames every 3s) to Python ML.
4. The **Python ML backend**:
   - Uses **Sliding Window Detector** with voting mechanism for continuous classification
   - OR **Hybrid Mode** combining motion-based segmentation + sliding window
   - Classifies gestures using a **GRU-based classifier**
   - Applies **intelligent flush mechanism** at batch boundaries to prevent word loss ⭐ NEW!
   - Stores predicted words in memory
5. When the special sign **"PUSH"** arrives:
   - Takes all saved words
   - Feeds them into a **sentence-polishing model** (Qwen LLM)
   - Returns a fully corrected English sentence
   - Resets memory

📥 Landmarks → 🔄 Batch Processing → 🤖 AI Detection → 💾 Buffer Flush → 📝 Natural English sentence

### 🆕 New Features (2025):

- ⚡ **Buffer Flush Mechanism** - Prevents losing last word at batch boundaries
- 🔀 **Hybrid Mode** - Combines motion-based segmentation with continuous classification
- 🗳️ **Voting System** - Sliding window with majority voting for stability
- 🎯 **Confidence Tuning** - Separate thresholds for normal detection vs. flush
- 📊 **Detailed Logging** - Track detection decisions, voting, and conflicts
- 🔧 **Optimized Parameters** - Pre-configured scenarios for speed vs. quality

---

# 2. Quick Setup: Get Running in 3 Steps

### Prerequisites

- Docker + Docker Compose  
- (Optional) Python 3.10+ if running manually  

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/SignSpeak.git
cd SignSpeak
```

---

## Step 2 — Add the AI Models

Place your models inside:

```
app_models/
   qwen2.5-1.5b-instruct-q4_k_m.gguf <- polishing model
```

These files are **not included in the repository** - check https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf for download.

---

## Step 3 — Run the API

### Development mode

```bash
docker compose -f docker-compose.yml up --build
```

Once running, you should see:

```
Uvicorn running on http://0.0.0.0:8000
```

---

# 3. How to Use the API (The Important Part!)

The main entry point for the MediaPipe frontend is:

```
POST /api/predict_landmarks
```

Each request corresponds to **one or multiple signs** and contains a list of frames.

The App always returns **LIST** of responses untill the last `end of the sentence` - `{"results": **responses}`. When `USE_SEGMENTATOR` = `False` there is always only 1 element in the list.

Warning - by default model will try and split uploaded landmarks into individual words - if that is unwanted, please see **config** section below.

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
Model still under development - with basic configuration in conf.pt
```

### Sentence Polisher (Qwen or T5)

```
app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```


---

# You’re Ready to Go 🎉

## 6. Config for Development

Right now this app is in its **development** phase - the vocabulary is very limited:

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


Additionally, current **Segmenter** is based on holistics and best fitting parameters for current dataset.
It may generate some additional loss because of that. To disable it, simply put `USE_SEGMENTATOR` to `False` in .env file.
Then app will not segment input and will await already segmented one - **so 1 word for inputted frames**.

### Segmenter config - /app/model_logic/utils/config.py
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

### General Config - .env file
| Variable             | Example Value                                             | Description                                                                                                                                                                                                  |
|----------------------| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `USE_SEGMENTATOR`    | `False`                                                   | Enables or disables the motion-based segmentator. <br>• `True` → system automatically detects word boundaries. <br>• `False` → each `/predict_landmarks` request is treated as one complete sign (one word). |
| `USE_SLIDING_WINDOW` | `True`                                                    | Enables sliding window detector with voting mechanism. <br>• `True` → continuous classification with stability checking. <br>• `False` → uses traditional segmenter only. |
| `USE_HYBRID_MODE`    | `False`                                                   | **NEW!** Combines both segmenter and sliding window intelligently. <br>• `True` → uses adaptive strategy to merge results from both detectors. <br>• Provides best accuracy by leveraging strengths of both approaches. |
| `SPECIAL_LABEL`   | `PUSH`                                                    | Special command label indicating the end of a sentence. When the classifier predicts `"PUSH"`, the backend generates the full sentence and resets internal memory.                                           |
| `MODEL_PATH`     | `/app/app/model_logic/utils/model_configs/conf(large).pt` | Path to the ASL classifier model (`.pt`). The backend loads this model during startup.                                                                                                                       |
| `POLISHING_MODEL_PATH` | `/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf`       | Path to the LLM used for sentence polishing (Qwen or another GGUF-based model).                                                                                                                              |
| `MIN_CONFIDENCE_THRESHOLD` | `0.75`                                            | Master confidence threshold - filters ALL detections. <br>• Lower = more detections (higher recall). <br>• Higher = fewer false positives (higher precision). <br>• **Suggest: 0.75 for production** |
| `USE_HYBRID_NORMALIZATION` | `True`                                            | **NEW!** Normalization method for hand landmarks. <br>• `False` → Wrist-centered (126 features). <br>• `True` → Hybrid with spatial context (138 features, better accuracy). <br>• **Note:** Model must be trained with corresponding method! |

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

```
Batch 1: [HELLO frames... HE]  ← incomplete at end
         ↓ Without flush: LOST
         ↓ With flush: Emits "HELLO" (confidence 0.55+)
         
Batch 2: [LLO frames... WORLD frames...]
         ↓ Continues normally
```

### Flush Configuration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_FLUSH_ON_BATCH_END` | `True` | **CRITICAL!** Enable flush at batch end.<br>• `True` → Prevents losing last word ✅<br>• `False` → Last word in batch is lost ❌<br>• **Recommendation:** Always `True` for real-time |
| `MIN_FRAMES_FOR_FLUSH` | `30` | Minimum frames in buffer before allowing flush.<br>• Too low (10-20) → May emit noise/fragments<br>• Too high (40+) → May not flush short gestures<br>• **Optimal:** 25-30 frames |
| `FLUSH_MIN_CONFIDENCE` | `0.55` | Lower confidence threshold for flushed words.<br>• Lower than `MIN_CONFIDENCE_THRESHOLD` (0.6-0.65)<br>• Catches uncertain last words that might be valid<br>• **Optimal:** 0.50-0.58 |

### Why Flush is Essential

**Without Flush (FORCE_FLUSH_ON_BATCH_END=False):**
```python
# Segmenter waits for 6 frames silence
# Batch ends with gesture still in progress
if self.silence_count >= 6:  # Never happens at batch end!
    return word  # Never executed
# Result: segments = [] → return [] → ZERO detections ❌
```

**With Flush (FORCE_FLUSH_ON_BATCH_END=True):**
```python
# At batch end, force emission
flushed_result = self.segmenter.flush_buffer()
if flushed_result is not None:
    # Classify buffered frames
    word, confidence = classify(flushed_result)
    if confidence >= FLUSH_MIN_CONFIDENCE:  # 0.55
        return word  # Success! ✅
```

### Flush Optimization Scenarios

#### Scenario A: Sliding Window Only (Fastest, Good Quality)
```python
USE_SLIDING_WINDOW = True
USE_HYBRID_MODE = False
MIN_CONFIDENCE_THRESHOLD = 0.65
SLIDING_WINDOW_VOTING_SIZE = 25        # Larger voting window
SLIDING_WINDOW_VOTE_THRESHOLD = 18     # 72% consensus
MIN_FRAMES_FOR_FLUSH = 30
FLUSH_MIN_CONFIDENCE = 0.55
```

#### Scenario B: Hybrid Mode (Best Quality, Slower)
```python
USE_HYBRID_MODE = True
MIN_CONFIDENCE_THRESHOLD = 0.70
SLIDING_WINDOW_VOTING_SIZE = 25
SLIDING_WINDOW_VOTE_THRESHOLD = 18
MIN_FRAMES_FOR_FLUSH = 25              # Lower for segmenter
FLUSH_MIN_CONFIDENCE = 0.50            # More lenient (hybrid has 2 sources)
HYBRID_SOLO_DETECTION_MULTIPLIER = 0.90
```

#### Scenario C: Balanced (Speed/Quality Compromise)
```python
USE_SLIDING_WINDOW = True
MIN_CONFIDENCE_THRESHOLD = 0.68
SLIDING_WINDOW_VOTING_SIZE = 22
SLIDING_WINDOW_VOTE_THRESHOLD = 16     # ~73%
MIN_FRAMES_FOR_FLUSH = 28
FLUSH_MIN_CONFIDENCE = 0.58
```

### Java Backend Frame Selection (Complementary Settings)

The Java backend (`FrameBufferService.java`) determines batch timing and size:

```java
@Scheduled(fixedRate = 3000)  // Process every 3 seconds (optimal)
// was: 5000 (5 seconds - too slow)

@Value("${frame.selection.count:45}")  // Send 45 frames per batch (optimal)
// was: 30 (too few for complete gestures)
```

**Optimal Java Backend Settings:**
- **Timer:** `3000ms` (3 seconds) - Frequent enough to catch gestures, not too aggressive
- **Frame count:** `45 frames` - Provides enough context for Sliding Window (needs 60 total, overlap helps)

**Why 45 frames?**
- Typical gesture: 30-50 frames
- Sliding Window buffer: 60 frames
- With 45 frames + overlap from previous batch → sufficient context for voting mechanism

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

2. **Update `FrameBufferService.java`:**
```java
@Scheduled(fixedRate = 3000)  // 3 seconds
@Value("${frame.selection.count:45}")  // 45 frames
```

3. **Monitor logs:**
```python
LOG_MINIMAL = True
LOG_VOTING = True  # See voting consensus
LOG_FILTERING = True  # See confidence filtering
```

### Troubleshooting Flush Issues

| Problem | Solution |
|---------|----------|
| Last word always missing | Set `FORCE_FLUSH_ON_BATCH_END=True` |
| Too many false positives at batch end | Increase `FLUSH_MIN_CONFIDENCE` to 0.60+ |
| Fragments/noise in output | Increase `MIN_FRAMES_FOR_FLUSH` to 35-40 |
| Short gestures not detected | Decrease `MIN_FRAMES_FOR_FLUSH` to 20-25 |
| Detections are delayed | Decrease Java `fixedRate` to 2000-3000ms |
| Buffer overflow errors | Increase Java `frame.selection.count` to 50-60 |

### Real-World Example: How Flush Works Step-by-Step

**Scenario:** User signs "HELLO" (3 seconds), pauses briefly (1 second), then signs "WORLD" (2 seconds)

#### Timeline with Current Architecture:

```
T=0.0s → 3.0s: User signs "HELLO"
├─ Frontend: MediaPipe detects hands → sends ~90 frames via WebSocket
├─ Java Backend: Buffers frames in ConcurrentLinkedQueue
└─ Timer hasn't triggered yet...

T=3.0s → 4.0s: User pauses (no hands visible)
├─ Frontend: MediaPipe detects NO hands → SENDS NOTHING
└─ Java Backend: Buffer still holds ~90 frames from "HELLO"

T=4.0s → 6.0s: User signs "WORLD"
├─ Frontend: Sends ~60 more frames
└─ Java Backend: Buffer now has ~150 frames total

T=5.0s: ⏰ TIMER TRIGGERS (fixedRate = 5000ms)
├─ Java Backend: processBuffer() executes
│   ├─ Total frames in buffer: ~150
│   ├─ Selects 30 evenly distributed frames
│   └─ Sends to Python ML via HTTP POST
│
├─ Python ML: PipelineManager.process(30 frames)
│   ├─ Sliding Window: Processes frames, fills voting buffer
│   │   ├─ Window 1-60: Detects "HELLO" (confidence 0.88)
│   │   ├─ Voting: 18/20 votes for "HELLO" → EMITS ✓
│   │   └─ Remaining frames: partial "WORLD" data
│   │
│   ├─ END OF BATCH REACHED
│   │
│   ├─ FLUSH TRIGGERED (FORCE_FLUSH_ON_BATCH_END=True)
│   │   ├─ Check voting buffer: 12/20 votes for "WORLD"
│   │   ├─ Below threshold (13) but close...
│   │   ├─ flush_buffer() forces emission
│   │   ├─ Confidence: 0.58 (above FLUSH_MIN_CONFIDENCE=0.55) ✓
│   │   └─ EMITS "WORLD" ✓
│   │
│   └─ Response: ["HELLO", "WORLD"]
│
└─ Java Backend: Clears buffer, continues...

T=10.0s: ⏰ NEXT TIMER
└─ Buffer is empty → skips processing
```

#### What Happens WITHOUT Flush (FORCE_FLUSH_ON_BATCH_END=False):

```
T=5.0s: TIMER TRIGGERS
├─ Python ML: PipelineManager.process(30 frames)
│   ├─ Sliding Window: Detects "HELLO" → EMITS ✓
│   ├─ Voting buffer has 12/20 votes for "WORLD" (below threshold)
│   ├─ END OF BATCH
│   ├─ NO FLUSH → voting buffer contents DISCARDED
│   └─ Response: ["HELLO"]  ← "WORLD" is LOST! ❌
│
└─ "WORLD" detection is gone forever
```

#### Key Observations:

1. **Timer is independent of gesture timing**
   - 5s timer can trigger mid-gesture
   - Batch boundaries are arbitrary, not natural pause points

2. **Frontend only sends frames with hands**
   - No "silence" frames between gestures
   - Segmenter can't detect word boundaries naturally

3. **Flush rescues incomplete detections**
   - Voting buffer at batch end may be close to threshold
   - Flush uses lower confidence to catch these

4. **30 frames is often insufficient**
   - "HELLO" gesture: ~40-50 frames at 30fps
   - 30 frames might only capture 60-70% of the gesture
   - Flush allows classification of partial data

### Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + MediaPipe)              │
│  • Captures webcam at ~30fps                                 │
│  • MediaPipe detects hand landmarks                          │
│  • Sends frames ONLY when hands detected                     │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket (continuous stream)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              JAVA BACKEND (Spring Boot + WebSocket)          │
│  • Receives frames via WebSocket                             │
│  • Buffers in ConcurrentLinkedQueue                          │
│  • @Scheduled(fixedRate=3000) triggers every 3s              │
│  • Selects 45 distributed frames from buffer                 │
│  • Sends batch to Python ML                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST (every 3s, 45 frames)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PYTHON ML (FastAPI + PyTorch)                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. PipelineManager receives batch                   │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Sliding Window Detector                          │   │
│  │     • Builds 60-frame windows (stride=1)             │   │
│  │     • Classifies each window → prediction            │   │
│  │     • Voting buffer (20 predictions)                 │   │
│  │     • Emits word when 18/20 agree                    │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. END OF BATCH - Flush Decision Point              │   │
│  │     IF FORCE_FLUSH_ON_BATCH_END = True:              │   │
│  │       • Check voting buffer                          │   │
│  │       • If word has 70% votes (flexible threshold)   │   │
│  │       • AND confidence >= FLUSH_MIN_CONFIDENCE       │   │
│  │       • EMIT word (prevents loss)                    │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  4. Word Buffer & Response                           │   │
│  │     • Accumulates words: ["HELLO", "WORLD"]          │   │
│  │     • Returns to Java Backend                        │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON Response
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND - Display Results                      │
│  • Shows detected words in real-time                         │
│  • User sees: "HELLO WORLD"                                  │
└─────────────────────────────────────────────────────────────┘
```

---

| `USE_T5`         | `False`                                                   | If `True`, the backend uses a T5 model for sentence polishing instead of Qwen. <br>Default: `False`.                                                                                                         |

---

# 🚀 Quick Reference Guide

## Most Important Settings for Production

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


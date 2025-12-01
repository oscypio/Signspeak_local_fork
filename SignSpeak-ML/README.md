# SignSpeak API: A Simple Guide

## Table of Contents

1. **What Does This Backend Do?**  

2. **Quick Setup: Get Running in 3 Steps**  

3. **How to Use the API (The Important Part!)**  
   - The Key Endpoint: `/api/predict_landmarks`
   - Example 1: Sending the sign **“NEED”**
   - Example 2: Sending the sign **“PHONE”**
   - Example 3: Ending the sentence with **“PUSH”**
   - Utility Endpoints: Force end & reset buffer
   - Full User Journey: A Realistic Conversation  
   
4. **Testing the API Yourself (Postman, Insomnia, or Code)**  

5. **Where Do the AI Models Come From?**

6. **Config for Development and Enhancements**

---

# 1. What Does This Backend Do?

Think of this backend as the **brain that understands sign language**.  
It doesn’t need video — all it needs is **hand landmarks**, the (x, y, z) coordinates of 21 points per hand, frame by frame.

Here is how the system works:

1. The MediaPipe frontend captures hand landmarks from a webcam.
2. For each sign (e.g., “HELLO”), the frontend collects all frames belonging to that sign.
3. It sends those frames to our backend through the `/api/process` endpoint.
4. The backend:
   - optionally performs segmentation,
   - predicts the sign using a GRU classifier,
   - stores the predicted word in memory.
5. When the special sign **“PUSH”** (or other) arrives:
   - it takes all saved words,
   - feeds them into a sentence-polishing model (Qwen or T5),
   - returns a fully corrected English sentence,
   - and resets memory.

📥 Landmarks → 🤖 AI pipeline → 📝 Natural English sentence

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

The configuration is now organized into **4 logical sections** in `/app/model_logic/utils/config.py`:

1. **Core Detection** - Segmenter & Sliding Window settings
2. **Hybrid Mode** - Adaptive combination parameters
3. **Model & Preprocessing** - Model paths, normalization, data preparation
4. **Performance** - Buffer management, flush settings

Each variable includes:
- 📝 Detailed description of what it does
- 💡 Suggested values for different use cases
- ⚠️ Warnings for critical parameters (e.g., DON'T CHANGE without retraining)
- 🗑️ DEPRECATED markers for obsolete settings
| `USE_T5`         | `False`                                                   | If `True`, the backend uses a T5 model for sentence polishing instead of Qwen. <br>Default: `False`.                                                                                                         |

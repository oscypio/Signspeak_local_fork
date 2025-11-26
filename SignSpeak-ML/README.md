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

Current state (when using conf(large).pt):

| Metric    | Value      |
| --------- |------------|
| Accuracy  | **82.42%** |
| Avg Loss  | **0.6356** |
| Macro F1  | **0.786**  |
| Precision | **0.823**  |
| Recall    | **0.800**  |

Glossary:

| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| BAD       | 0.800     | 0.800  | 0.800    |
| GOOD      | 1.000     | 0.200  | 0.333    |
| HELLO     | 0.857     | 1.000  | 0.923    |
| HOW       | 1.000     | 1.000  | 1.000    |
| LUNCH     | 0.800     | 0.800  | 0.800    |
| MUCH      | 0.571     | 0.800  | 0.667    |
| NEED      | 1.000     | 1.000  | 1.000    |
| PHONE     | 0.833     | 1.000  | 0.909    |
| PLEASE    | 0.833     | 1.000  | 0.909    |
| PUSH      | 1.000     | 1.000  | 1.000    |
| THANK YOU | 0.400     | 0.400  | 0.400    |
| WAIT      | 1.000     | 0.600  | 0.750    |
| WANT      | 0.600     | 0.600  | 0.600    |

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
| `SPECIAL_LABEL`   | `PUSH`                                                    | Special command label indicating the end of a sentence. When the classifier predicts `"PUSH"`, the backend generates the full sentence and resets internal memory.                                           |
| `MODEL_PATH`     | `/app/app/model_logic/utils/model_configs/conf(large).pt` | Path to the ASL classifier model (`.pt`). The backend loads this model during startup.                                                                                                                       |
| `POLISHING_MODEL_PATH` | `/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf`       | Path to the LLM used for sentence polishing (Qwen or another GGUF-based model).                                                                                                                              |
| `USE_T5`         | `False`                                                   | If `True`, the backend uses a T5 model for sentence polishing instead of Qwen. <br>Default: `False`.                                                                                                         |

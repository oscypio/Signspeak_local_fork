import os
import time
import json
import pickle
import urllib.request
import urllib.error
import numpy as np
import re
from typing import List, Dict, Any, Tuple


def http_post_json(url: str, payload: Any, timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


# --- helpers copied/adapted from test ---
def load_pkl_to_frames(pkl_path: str) -> List[Dict[str, Any]]:
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)

    keypoints = np.array(data.get('keypoints'))
    T = keypoints.shape[0]
    if keypoints.ndim == 3 and keypoints.shape[1] == 42:
        keypoints = keypoints.reshape(T, 2, 21, -1)

    timestamps = data.get('timestamps')
    if timestamps is None:
        timestamps = np.arange(T, dtype=float) * 0.033
    else:
        timestamps = np.array(timestamps, dtype=float)

    frames = []
    for i in range(T):
        frame_kp = keypoints[i]
        hands_landmarks = []
        hands_handedness = []
        for hand_id, hand_points in enumerate(frame_kp):
            lm_list = []
            for pt in hand_points:
                x = float(pt[0])
                y = float(pt[1])
                z = float(pt[2]) if len(pt) > 2 else 0.0
                vis = float(pt[3]) if len(pt) > 3 else 1.0
                lm_list.append({"x": x, "y": y, "z": z, "visibility": vis})
            hands_landmarks.append(lm_list)
            handedness_label = "Right" if hand_id == 0 else "Left"
            hands_handedness.append([{"score": 1.0, "index": int(hand_id), "categoryName": handedness_label, "displayName": handedness_label}])

        frames.append({
            "timestamp": float(timestamps[i]),
            "sequenceNumber": int(i),
            "receivedAt": float(timestamps[i]),
            "landmarks": hands_landmarks,
            "handedness": hands_handedness,
        })

    return frames


def extract_texts_from_resp(resp: Any) -> List[str]:
    """
    Extract all text predictions from API response.
    For SignSpeak-ML API, this includes:
    - 'prediction' fields (individual words)
    - 'sentence' fields (completed sentences)
    - 'current_words' lists (accumulated words)
    """
    texts = []

    if isinstance(resp, dict):
        # Check for 'results' list (standard SignSpeak-ML format)
        res = resp.get('results')
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    # Extract 'prediction' (individual word)
                    pred = item.get('prediction')
                    if isinstance(pred, str) and pred.strip():
                        texts.append(pred.strip())

                    # Extract 'sentence' (completed sentence from end_of_sentence status)
                    sent = item.get('sentence')
                    if isinstance(sent, str) and sent.strip():
                        texts.append(sent.strip())

                    # Extract from 'current_words' list
                    curr = item.get('current_words')
                    if isinstance(curr, list):
                        for w in curr:
                            if isinstance(w, str) and w.strip():
                                texts.append(w.strip())
                    elif isinstance(curr, str) and curr.strip():
                        texts.append(curr.strip())

        # Fallback: check top-level keys
        for key in ("sentence", "text", "transcription", "prediction", "predicted_text", "decoded"):
            v = resp.get(key)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())

    elif isinstance(resp, list):
        for item in resp:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
            elif isinstance(item, dict):
                # Recursively extract from dict items
                sub_texts = extract_texts_from_resp(item)
                texts.extend(sub_texts)

    elif isinstance(resp, str) and resp.strip():
        texts.append(resp.strip())

    # deduplicate preserving order
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def evaluate_pkl_dir(pkl_dir: str,
                     ml_base_url: str = "http://localhost:8000",
                     timeout: float = 120.0,
                     print_summary: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate all .pkl files in `pkl_dir` by sending them to the SignSpeak-ML API.

    Behaviour (requirements):
    - For each .pkl file send frames to POST /api/predict_landmarks
    - For each response compute: response_time_s, matches, extras, returned_sentence
      * returned_sentence prefers explicit JSON field 'sentence' (searched top-level, then anywhere inside results)
      * fallback to the first string extracted by extract_texts_from_resp
    - Print per-file lines: time, matches, extras, returned
    - Aggregate statistics only over successful responses (those with response_time_s)
    - At the end POST to /api/reset_buffer to clear model buffer

    Returns (per_file_stats, summary)
    """
    # Validate input directory
    if not os.path.isdir(pkl_dir):
        raise FileNotFoundError(f"PKL directory not found: {pkl_dir}")

    # Collect .pkl files
    pkl_files = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.lower().endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")

    predict_url = f"{ml_base_url.rstrip('/')}/api/predict_landmarks"
    reset_url = f"{ml_base_url.rstrip('/')}/api/reset_buffer"

    per_file_stats: List[Dict[str, Any]] = []

    for pkl_path in pkl_files:
        fname = os.path.basename(pkl_path)
        expected_stem = os.path.splitext(fname)[0]
        expected_words = [tok.strip().upper() for tok in re.split(r'[^A-Za-z0-9]+', expected_stem) if tok.strip()]
        expected_set = set(expected_words)

        # Load frames
        try:
            frames = load_pkl_to_frames(pkl_path)
        except Exception as e:
            per_file_stats.append({'file': fname, 'error': f'failed to load pkl: {e}'})
            continue

        # Send to API and measure time
        start = time.time()
        try:
            resp = http_post_json(predict_url, frames, timeout=timeout)
        except Exception as e:
            per_file_stats.append({'file': fname, 'error': f'http error: {e}'})
            continue
        elapsed = time.time() - start

        # Extract textual outputs
        texts = extract_texts_from_resp(resp)

        # Extract sentence from response (prefer 'sentence' from end_of_sentence status)
        sentence_from_json = ""
        predictions_list = []
        final_words_list = []
        all_returned_words = []  # NEW: Track all words returned by API in order

        if isinstance(resp, dict):
            res = resp.get('results')
            if isinstance(res, list):
                # Search for 'sentence' in objects with status='end_of_sentence'
                for itm in res:
                    if isinstance(itm, dict):
                        status = itm.get('status', '')

                        # Priority 1: sentence from end_of_sentence
                        if status == 'end_of_sentence':
                            s_itm = itm.get('sentence')
                            if isinstance(s_itm, str) and s_itm.strip():
                                sentence_from_json = s_itm.strip()
                                # Don't break - continue collecting all words

                        # Collect predictions for fallback AND for all_returned_words
                        pred = itm.get('prediction')
                        if isinstance(pred, str) and pred.strip():
                            word = pred.strip()
                            predictions_list.append(word)
                            # Add to all_returned_words if not SPECIAL_LABEL (unless it's end_of_sentence)
                            if status != 'end_of_sentence':
                                all_returned_words.append(word)

                        # Collect current_words from last item
                        curr = itm.get('current_words')
                        if isinstance(curr, list):
                            final_words_list = [str(w).strip() for w in curr if str(w).strip()]

                # Priority 2: if no sentence, build from last current_words
                if not sentence_from_json and final_words_list:
                    sentence_from_json = " ".join(final_words_list)

                # Priority 3: if no current_words, build from predictions
                if not sentence_from_json and predictions_list:
                    sentence_from_json = " ".join(predictions_list)

            # Fallback: top-level sentence
            if not sentence_from_json:
                s_top = resp.get('sentence')
                if isinstance(s_top, str) and s_top.strip():
                    sentence_from_json = s_top.strip()

        # If no words collected from results, use final_words_list or predictions_list
        if not all_returned_words:
            if final_words_list:
                all_returned_words = final_words_list
            elif predictions_list:
                all_returned_words = predictions_list

        # Build normalized response words for matching (uppercase alnum tokens)
        response_words: List[str] = []
        for t in texts:
            for w in t.split():
                cleaned = re.sub(r"[^A-Za-z0-9]", "", w).upper()
                if cleaned:
                    response_words.append(cleaned)
        response_set = set(response_words)

        # Compute matches/extras
        matches = sum(1 for w in expected_set if w in response_set)
        extras = sum(1 for w in response_set if w not in expected_set)

        # Basic classification metrics (per-file)
        recall = matches / len(expected_words) if expected_words else 0.0
        precision = matches / (matches + extras) if (matches + extras) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Choose returned sentence: explicit JSON 'sentence' preferred, then first extracted text
        returned_sentence = sentence_from_json if sentence_from_json else (texts[0] if texts else "")

        per_file_stats.append({
            'file': fname,
            'expected': expected_words,
            'found_texts': texts,
            'returned_sentence': returned_sentence,
            'response_words': response_words,
            'all_returned_words': all_returned_words,  # NEW: Add all returned words to stats
            'response_time_s': elapsed,
            'detected_words_count': len(response_words),
            'matches': matches,
            'extras': extras,
            'recall': recall,
            'precision': precision,
            'f1': f1,
        })

        # NEW: Reset buffer after each file to ensure clean state for next test
        try:
            reset_resp = http_post_json(reset_url, {}, timeout=5.0)
            if print_summary:
                print(f"  → Buffer reset for next file")
        except Exception as e:
            if print_summary:
                print(f"  → Warning: Failed to reset buffer: {e}")

    # Aggregate statistics only over successful entries
    total = len(per_file_stats)
    successful_entries = [p for p in per_file_stats if 'response_time_s' in p and 'error' not in p]
    success_count = len(successful_entries)

    if success_count > 0:
        avg_time = sum(p['response_time_s'] for p in successful_entries) / success_count
        avg_recall = sum(p.get('recall', 0.0) for p in successful_entries) / success_count
        avg_precision = sum(p.get('precision', 0.0) for p in successful_entries) / success_count
        avg_f1 = sum(p.get('f1', 0.0) for p in successful_entries) / success_count
    else:
        avg_time = avg_recall = avg_precision = avg_f1 = 0.0

    summary = {
        'total': total,
        'successful': success_count,
        'avg_time_s': avg_time,
        'avg_recall': avg_recall,
        'avg_precision': avg_precision,
        'avg_f1': avg_f1,
    }

    # Print per-file summary lines (restricted fields) and short aggregate
    if print_summary:
        print('\n=== PKL Batch Evaluation: per-file results ===')
        for p in per_file_stats:
            if 'error' in p:
                print(f"- {p['file']}: ERROR: {p['error']}")
            else:
                words_str = ' '.join(p.get('all_returned_words', [])) if p.get('all_returned_words') else '(none)'
                print(f"- {p['file']}: time={p.get('response_time_s'):.3f}s, matches={p.get('matches')}, extras={p.get('extras')}")
                print(f"    returned_words: [{words_str}]")
                print(f"    returned_sentence: '{p.get('returned_sentence')}'")
        print('\n=== Aggregated (successful requests) ===')
        print(f"Successful: {success_count}/{total}, avg_time_s: {avg_time:.3f}, avg_recall: {avg_recall:.3f}, avg_precision: {avg_precision:.3f}, avg_f1: {avg_f1:.3f}")

    # Reset model buffer at the end
    try:
        reset_resp = http_post_json(reset_url, {}, timeout=timeout)
        if print_summary:
            print(f"Buffer reset response: {reset_resp}")
    except Exception as e:
        if print_summary:
            print(f"Failed to reset buffer: {e}")

    return per_file_stats, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate directory of PKL files against SignSpeak-ML API")
    parser.add_argument("pkl_dir", nargs="?", default=os.environ.get('PKL_TEST_DIR', 'C:/Users/Michal/Documents/Studia/Studia_Rok_3/SA/src/data/ASL_Citizen/sel_vid/demo_tests'), help="Directory with .pkl files")
    parser.add_argument("--ml-base-url", default=os.environ.get('ML_BASE_URL', 'http://localhost:8000'), help="ML API base URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument("--no-print", dest="print_summary", action="store_false", help="Don't print summary")

    args = parser.parse_args()
    if not args.pkl_dir:
        print("Error: pkl_dir not provided and PKL_TEST_DIR env not set")
        raise SystemExit(2)

    stats, summary = evaluate_pkl_dir(args.pkl_dir, ml_base_url=args.ml_base_url, timeout=args.timeout, print_summary=args.print_summary)
    # Optionally save summary to file
    out_path = os.path.join(args.pkl_dir, 'pkl_batch_summary.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'per_file': stats}, fh, indent=2)
        if args.print_summary:
            print(f"Wrote summary to {out_path}")
    except Exception:
        pass

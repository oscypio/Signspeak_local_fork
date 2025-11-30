from collections import Counter

import torch
import numpy as np
from typing import Optional, Dict, Type, cast

from .models.GRUClassifier import GRUClassifier
from ..utils.config import settings


class ASLClassifier:
    """
    Production-ready wrapper for any ASL recognition model.
    Works as a stable interface between Pipeline and the underlying ML model.

    Responsibilities:
      - Load model from file
      - Keep class mappings
      - Prepare input tensors
      - Run inference (predict index / label / probabilities)
      - Allow easy model swapping (GRU, Transformer, CNN, etc.)
    """

    def __init__(
        self,
        model_path: str = settings.MODEL_PATH,
        device: str = settings.DEVICE,
        model_class: Optional[Type[torch.nn.Module]] = None,
        expected_frames: int = settings.EXPECTED_FRAMES,
    ):
        """
        Args:
            model_path: path to saved model .pt
            device: "cpu" or "cuda"
            model_class: override model type (default = GRUClassifier)
            expected_frames: fixed length for resampling inputs
        """
        self.device = device
        self.expected_frames = expected_frames

        # --- Support future model types ---
        self.model_class = model_class or GRUClassifier

        # --- Load model ---
        self.model = self.model_class.from_file(model_path, device=device)
        self.model.eval()

        # --- Metadata ---
        self.class_names = self.model.class_names
        self.class_to_idx = self.model.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    # ------------------------------------------------------------------
    # Internal helper — Convert (T,F) np.ndarray → torch tensor (1,T,F)
    # ------------------------------------------------------------------

    def _prepare_tensor(self, seq_np: np.ndarray) -> torch.Tensor:
        """
        Converts np.ndarray(T,F) to tensor(1,T,F) and moves to device.
        Assumes seq_np is already normalized and has correct shape.
        """
        if not isinstance(seq_np, np.ndarray):
            seq_np = np.array(seq_np, dtype=np.float32)

        x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_index(self, seq_np: np.ndarray) -> int:
        """
        Returns the class index (int).
        """
        x = self._prepare_tensor(seq_np)
        with torch.no_grad():
            logits = self.model(x)
            idx = torch.argmax(logits, dim=1).item()
        return idx

    def predict_label(self, seq_np: np.ndarray) -> str:
        """
        Returns the predicted class label (str).
        """
        idx = self.predict_index(seq_np)
        return self.idx_to_class[idx]

    def predict_proba(self, seq_np: np.ndarray) -> Dict[str, float]:
        """
        Returns {label: probability}.
        """
        x = self._prepare_tensor(seq_np)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        labels = [self.idx_to_class[int(i)] for i in range(len(probs))]
        return {label: float(p) for label, p in zip(labels, probs)}

    def predict_proba_batch(self, seq_list: list[np.ndarray]) -> list[Dict[str, float]]:
        """
        Batch prediction for multiple sequences.

        Input:
            seq_list: list of np.ndarray each with shape (T, F)
        Returns:
            list of dicts {label:prob} in the same order as input
        """
        if not seq_list:
            return []

        # Prepare a single tensor (N, T, F)
        tensors = []
        for seq in seq_list:
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            tensors.append(torch.tensor(seq, dtype=torch.float32))

        # Stack and move to device
        x = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)  # (N, C)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, C)

        out = []
        for row in probs:
            labels = [self.idx_to_class[int(i)] for i in range(len(row))]
            out.append({label: float(p) for label, p in zip(labels, row)})

        return out

    def predict_best_from_candidates(self, candidates: list[np.ndarray], *,
                                     scoring_method: str = None,
                                     early_stop: float = None,
                                     batch_predict: bool = None,
                                     return_confidence: bool = False):
        """
        Given a list of candidate segments (np.ndarray (T,F)), returns the
        most probable class label according to scoring_method.

        Parameters:
            candidates: list of np.ndarray segments
            scoring_method: override settings.SEGMENTER_SCORING_METHOD
            early_stop: override settings.SEGMENTER_EARLY_STOP_PROB
            batch_predict: override settings.SEGMENTER_BATCH_PREDICT
            return_confidence: if True, returns (label, confidence) tuple instead of just label

        Returns:
            label (str) or None if candidates empty
            OR (label, confidence) tuple if return_confidence=True
        """
        if not candidates:
            if return_confidence:
                return (None, 0.0)
            return cast(Optional[str], None)

        scoring_method = scoring_method or settings.SEGMENTER_SCORING_METHOD
        early_stop = early_stop if early_stop is not None else settings.SEGMENTER_EARLY_STOP_PROB
        batch_predict = batch_predict if batch_predict is not None else settings.SEGMENTER_BATCH_PREDICT


        resampled = [np.array(seq, dtype=np.float32) for seq in candidates]

        # Get probability dictionaries
        if batch_predict:
            probs_list = self.predict_proba_batch(resampled)
        else:
            probs_list = [self.predict_proba(seq) for seq in resampled]


        # New aggregate scoring: sum probabilities per label across all
        # candidates and pick the label with highest total probability.
        if scoring_method == 'sum_label_probs':
            totals: Dict[str, float] = {}
            for probs in probs_list:
                for label, p in probs.items():
                    totals[label] = totals.get(label, 0.0) + float(p)

            if not totals:
                if return_confidence:
                    return (None, 0.0)
                return cast(Optional[str], None)

            # return label with maximum aggregated probability
            best_label, total_prob = max(totals.items(), key=lambda kv: kv[1])

            if return_confidence:
                # Normalize confidence by number of candidates
                confidence = min(total_prob / len(candidates), 1.0)
                return (best_label, confidence)

            return best_label

        # Fallback: evaluate candidates individually and pick best by chosen
        # per-candidate scoring method (existing behavior)
        best_score = -float('inf')
        best_label = None

        for probs in probs_list:
            if not probs:
                score = -float('inf')
            else:
                vals = list(probs.values())
                if scoring_method == 'sum_prob':
                    score = sum(vals)
                elif scoring_method == 'mean_prob':
                    score = float(sum(vals)) / len(vals) if vals else 0.0
                elif scoring_method == 'neg_entropy':
                    # negative entropy (higher is better)
                    probs_arr = np.array(vals, dtype=np.float64)
                    probs_arr = np.clip(probs_arr, 1e-12, 1.0)
                    ent = -np.sum(probs_arr * np.log(probs_arr))
                    score = -ent
                else:
                    # default: max probability
                    score = max(vals)

            if score > best_score:
                best_score = score
                # choose label with highest probability in this candidate
                best_label = max(probs.items(), key=lambda kv: kv[1])[0]

            if early_stop is not None and best_score >= early_stop:
                break

        if return_confidence:
            # Normalize score to 0-1 range if needed
            confidence = min(max(best_score, 0.0), 1.0) if best_score != -float('inf') else 0.0
            return (best_label, confidence)

        return best_label

    # ------------------------------------------------------------------
    # Convenience method for pipeline
    # ------------------------------------------------------------------

    def predict(self, seq_np: np.ndarray) -> str:
        """
        Standard prediction for Pipeline:
        returns a class label (str).
        """
        return self.predict_label(seq_np)

    def predict_tta(self, tta_segments: list[np.ndarray], return_confidence: bool = False):
        """
        TTA inference:
        - tta_segments: list of (60,F) numpy arrays generated by DataPreparer
        - returns: final predicted label (str) after majority vote
                   OR (label, confidence) tuple if return_confidence=True
        """
        if not tta_segments:
            raise ValueError("predict_tta received empty segment list")

        preds = []

        for seq_np in tta_segments:
            # Standard single-sequence prediction
            label = self.predict(seq_np)  # already returns a str
            preds.append(label)

        # Majority vote (mode)
        counter = Counter(preds)
        final_label, vote_count = counter.most_common(1)[0]

        if return_confidence:
            # Confidence = voting ratio (how many agreed on this label)
            confidence = vote_count / len(tta_segments)
            return (final_label, confidence)

        return final_label

    # --- optional: return all predictions for debugging ---
    def predict_tta_debug(self, tta_segments: list[np.ndarray]):
        """
        Returns:
            {
                'votes': {'HELLO': 5, 'PHONE': 2},
                'final': 'HELLO',
                'raw_preds': ['HELLO','HELLO','PHONE',...]
            }
        """
        if not tta_segments:
            return {
                "votes": {},
                "final": None,
                "raw_preds": []
            }

        preds = [self.predict(seq) for seq in tta_segments]
        counter = Counter(preds)
        final = counter.most_common(1)[0][0]

        return {
            "votes": dict(counter),
            "final": final,
            "raw_preds": preds
        }

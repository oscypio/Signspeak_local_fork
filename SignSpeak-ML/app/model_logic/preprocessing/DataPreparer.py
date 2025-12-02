import numpy as np
import torch
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

from ..utils.config import settings


# ============================================================
#     UNIFIED DATA PREPARER FOR TRAINING + RUNTIME
# ============================================================

class UnifiedDataPreparer:
    """
    Single consistent preprocessing class used in BOTH training and runtime.
    Handles:
        - loading .pkl files (training)
        - processing FrameData objects (runtime)
        - normalization
        - flattening
        - optional velocity & acceleration
        - fixed-length resampling
    """

    def __init__(
        self,
        target_frames: int = settings.EXPECTED_FRAMES,
        add_feat: bool = settings.ADD_FEATURES,    # velocity + acceleration
        flatten: bool = True
    ):
        self.target_frames = target_frames
        self.add_feat = add_feat
        self.flatten = flatten


    def prepare_from_pkl_list(self, pkl_paths: List[Path]):
        sequences = []
        lengths = []
        labels = []

        for path in pkl_paths:
            seq, label = self._load_single_pkl(path)
            sequences.append(seq)
            lengths.append(self.target_frames)
            labels.append(label)

        x = torch.stack(sequences, dim=0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        input_size = x.shape[-1]

        return (x, lengths, labels), input_size

    def prepare_from_api(self, frames: list) -> np.ndarray:
        """
        Processes list[FrameData] from runtime API.
        Returns:
            np.ndarray of shape (target_frames, F)
        """
        if not frames:
            return None

        coords = self._extract_raw_coords_from_api(frames)

        coords = self.normalize_landmarks(coords)  # (T, 42, 4)

        coords = coords.reshape(coords.shape[0], -1)  # (T, 168)

        seq = self._process_sequence(coords)  # (target_frames, F)

        return seq

    def prepare_raw(self, frames):
        coords = self._extract_raw_coords_from_api(frames)

        # Choose normalization method based on config
        if settings.USE_HYBRID_NORMALIZATION:
            coords = self.normalize_landmarks_hybrid(coords)
        else:
            coords = self.normalize_landmarks(coords)

        coords = coords.reshape(coords.shape[0], -1)
        return coords  # (T, F) ORIGINAL FPS

    def prepare_resampled(self, seq):
        T = seq.shape[0]
        if T == self.target_frames:
            return seq
        idx = np.linspace(0, T - 1, self.target_frames, dtype=int)
        return seq[idx]

    def prepare_tta_segments(self, segment_raw: np.ndarray, n_augs: int = 7):
        """
        Input:
            segment_raw = (T, F) raw segment from segmenter

        Returns:
            List[np.ndarray]  — list of (60, F) TTA-resampled sequences
        """

        tta_sequences = []

        # 1) Base variant (no augmentation)
        base_resampled = self.prepare_resampled(segment_raw)
        tta_sequences.append(base_resampled)

        # 2) Augmented variants
        for _ in range(n_augs):
            aug_raw = self._augment_raw_segment(segment_raw)
            aug_resampled = self.prepare_resampled(aug_raw)
            tta_sequences.append(aug_resampled)

        return tta_sequences

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _augment_raw_segment(self, seq: np.ndarray) -> np.ndarray:
        """
        Applies temporal augmentations BEFORE resampling.
        seq: (T, F)
        """
        T = seq.shape[0]

        # ---- 1. Random crop/zoom ----
        crop_ratio = np.random.uniform(0.85, 1.0)
        crop_len = max(8, int(T * crop_ratio))
        start = np.random.randint(0, max(1, T - crop_len))
        seq_aug = seq[start:start + crop_len]

        # ---- 2. Temporal shift ----
        shift = np.random.randint(-2, 3)
        if shift > 0:
            seq_aug = seq_aug[shift:]
        elif shift < 0:
            pad = np.repeat(seq_aug[:1], -shift, axis=0)
            seq_aug = np.concatenate([pad, seq_aug], axis=0)

        # ---- 3. Optional jitter (drop or duplicate one frame) ----
        if np.random.rand() < 0.25 and len(seq_aug) > 10:
            drop_i = np.random.randint(1, len(seq_aug) - 1)
            seq_aug = np.delete(seq_aug, drop_i, axis=0)

        if np.random.rand() < 0.25:
            dup_i = np.random.randint(0, len(seq_aug))
            seq_aug = np.insert(seq_aug, dup_i, seq_aug[dup_i], axis=0)

        return seq_aug

    def _load_single_pkl(self, path: Path):
        """
        Loads single PKL (training).
        Returns: (tensor(T,F), label_string)
        """
        with open(path, "rb") as f:
            pkl = pickle.load(f)

        raw = np.array(pkl["keypoints"], dtype=np.float32)  # (T,42,4)

        # Choose normalization method based on config
        if settings.USE_HYBRID_NORMALIZATION:
            raw = self.normalize_landmarks_hybrid(raw)
        else:
            raw = self.normalize_landmarks(raw)

        raw = raw.reshape(raw.shape[0], -1)  # (T,168) or (T,138) for hybrid

        seq = self._process_sequence(raw)
        label = self._normalize_label(pkl.get("label", ""))

        return torch.tensor(seq, dtype=torch.float32), label

    # --------------------------------------------------------

    def _extract_raw_coords_from_api(self, frames):
        """
        Converts runtime API FrameData list into np.ndarray (T,42,4)
        """
        T = len(frames)
        out = np.zeros((T, 42, 4), dtype=np.float32)

        for i, frame in enumerate(frames):
            idx = 0
            for hand in frame.landmarks:
                for lm in hand:
                    out[i, idx, 0] = lm.x
                    out[i, idx, 1] = lm.y
                    out[i, idx, 2] = lm.z
                    out[i, idx, 3] = lm.visibility if hasattr(lm, "visibility") else 1.0
                    idx += 1

            # If one hand missing → remaining 21 keypoints = zeros

        return out

    # --------------------------------------------------------

    def _process_sequence(self, arr: np.ndarray) -> np.ndarray:
        """
        Takes (T, F_raw) in normalized float format.
        Adds features, resamples, flattens.
        Returns: (target_frames, F_final)
        """
        T = arr.shape[0]

        # Optional velocity / acceleration
        if self.add_feat and T > 1:
            vel = np.diff(arr, axis=0, prepend=arr[:1])
            acc = np.diff(vel, axis=0, prepend=vel[:1])
            arr = np.concatenate([arr, vel, acc], axis=-1)

        # Resampling to fixed length
        if T == self.target_frames:
            arr_fixed = arr
        else:
            idx = np.linspace(0, T - 1, self.target_frames, dtype=int)
            arr_fixed = arr[idx]

        return arr_fixed.astype(np.float32)

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Cleans label names, e.g. 'LIVE2' → 'LIVE'."""
        return re.sub(r"[_\-\s]*\d+$", "", str(label).upper())

    @staticmethod
    def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
        """
        Identical normalization for training + runtime.
        Wrist-centered, hand-size scaled.
        """
        original_shape = coords.shape

        # Detect format
        if coords.ndim == 3 and coords.shape[1] == 42:
            T = coords.shape[0]
            coords = coords.copy()
            flat = False
        else:
            T = coords.shape[0]
            feat_dim = coords.shape[1]

            if feat_dim == 42 * 4:
                coords = coords.reshape(T, 42, 4).copy()
            elif feat_dim == 42 * 3:
                coords = coords.reshape(T, 42, 3).copy()
            else:
                raise ValueError(f"Unsupported landmark format: {coords.shape}")

            flat = True

        if coords.shape[2] == 4:
            xyz = coords[..., :3]
            vis = coords[..., 3:]
        else:
            xyz = coords
            vis = None

        xyz_hands = xyz.reshape(T, 2, 21, 3)
        vis_hands = vis.reshape(T, 2, 21, 1) if vis is not None else None

        out_xyz = np.zeros_like(xyz_hands, dtype=np.float32)

        for t in range(T):
            for h in range(2):
                hand = xyz_hands[t, h]
                if not hand.any():
                    continue

                wrist = hand[0]
                centered = hand - wrist

                scale = np.linalg.norm(centered[9])
                if scale > 1e-6:
                    centered = centered / scale

                out_xyz[t, h] = centered

        if vis_hands is not None:
            out = np.concatenate([out_xyz, vis_hands], axis=3)
        else:
            out = out_xyz

        coords_out = out.reshape(T, 42, -1)

        if flat:
            coords_out = coords_out.reshape(T, -1)

        return coords_out.astype(np.float32)

    @staticmethod
    def normalize_landmarks_hybrid(coords: np.ndarray) -> np.ndarray:
        """
        Hybrid normalization - preserves spatial context.

        Combines:
        1. Wrist-centered coordinates (local hand shape) - 21 landmarks × 3 coords = 63 features
        2. Absolute wrist position (global context) - 3 features (x, y, z)
        3. Hand orientation vector (pointing direction) - 3 features

        Total: 69 features per hand × 2 hands = 138 features (vs 126 in wrist-centered)

        Benefits:
        - Preserves height information (y position)
        - Preserves depth information (z position)
        - Preserves horizontal position (x position)
        - Captures hand pointing direction
        - Still normalizes hand shape by size

        Args:
            coords: Input coordinates (T, 42, 4) or (T, 168) or (T, 126)

        Returns:
            Normalized coords with spatial context (T, 138) if flat, else (T, 2, 69)
        """
        original_shape = coords.shape

        # Detect format
        if coords.ndim == 3 and coords.shape[1] == 42:
            T = coords.shape[0]
            coords = coords.copy()
            flat = False
        else:
            T = coords.shape[0]
            feat_dim = coords.shape[1]

            if feat_dim == 42 * 4:
                coords = coords.reshape(T, 42, 4).copy()
            elif feat_dim == 42 * 3:
                coords = coords.reshape(T, 42, 3).copy()
            else:
                raise ValueError(f"Unsupported landmark format: {coords.shape}")

            flat = True

        # Extract xyz (drop visibility if present)
        if coords.shape[2] == 4:
            xyz = coords[..., :3]
        else:
            xyz = coords

        xyz_hands = xyz.reshape(T, 2, 21, 3)

        # Output: (T, 2, 69) - 69 features per hand
        # 63 (wrist-centered shape) + 3 (wrist position) + 3 (orientation)
        out_features = np.zeros((T, 2, 69), dtype=np.float32)

        for t in range(T):
            for h in range(2):
                hand = xyz_hands[t, h]

                # Check if hand is empty (all zeros)
                if not hand.any():
                    # Keep as zeros
                    continue

                wrist = hand[0]  # Landmark 0 - wrist
                middle_finger_mcp = hand[9]  # Landmark 9 - middle finger MCP
                middle_finger_tip = hand[12]  # Landmark 12 - middle finger tip

                # 1. Wrist-centered local shape (same as original)
                centered = hand - wrist
                scale = np.linalg.norm(centered[9])  # Distance to middle finger MCP

                if scale > 1e-6:
                    centered = centered / scale
                else:
                    # Very small hand or bad detection - use unscaled
                    centered = hand - wrist

                # 2. Absolute wrist position (global context)
                wrist_pos = wrist.copy()

                # 3. Hand orientation vector (normalized pointing direction)
                if scale > 1e-6:
                    # Direction from wrist to middle finger tip
                    orientation = (middle_finger_tip - wrist) / scale
                else:
                    # Default orientation if hand is too small
                    orientation = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                # Concatenate all features: [63 shape] + [3 wrist_pos] + [3 orientation] = 69
                out_features[t, h, :63] = centered.flatten()
                out_features[t, h, 63:66] = wrist_pos
                out_features[t, h, 66:69] = orientation

        # Reshape to (T, 138) if flat was True
        if flat:
            coords_out = out_features.reshape(T, -1)
        else:
            coords_out = out_features

        return coords_out.astype(np.float32)

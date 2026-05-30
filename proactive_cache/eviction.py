"""
eviction.py — Token scoring and KV cache pruning.

Core O(n) eviction policy:
  1. Score each token position using offline-profiled prototype centroids.
  2. Keep the top-budget tokens (attention sink + recency anchors + semantic prototypes).
  3. Prune the KV cache to exactly `budget` positions.

This is coordinate-free and RoPE-compatible: we only select positions, never
reorder them, so relative position encodings remain valid.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Dict, Tuple, List

from .utils import to_tuple_kv, to_dynamic_cache


def score_tokens(
    prototypes: Optional[Dict],
    seq_len: int,
    budget: int,
) -> np.ndarray:
    """
    Score all token positions using prototype centroid histograms.

    Algorithm (O(n) per call):
      - For each profiled (layer, head), accumulate the centroid attention
        histogram as a distance-weighted score over token positions.
      - Boost attention sink (token 0) unconditionally.
      - Boost a proportional recency window at the tail.
      - Add a small deterministic tiebreaker (position index).

    Args:
        prototypes: Output of ``build_prototypes()``. If None, falls back to
                    uniform scoring (no-op — keep all tokens equally).
        seq_len:    Current sequence length to score.
        budget:     Target number of tokens to keep.

    Returns:
        scores: (seq_len,) float64 array. Higher = more important.
    """
    scores = np.zeros(seq_len, dtype=np.float64)

    if prototypes is not None:
        for (layer, head), data in prototypes.items():
            centroid = data["centroids"][0]            # shape: (profile_seq_len,)
            max_d = min(len(centroid), seq_len)
            if max_d == 0:
                continue
            cumsum = np.cumsum(centroid[:max_d])
            for p in range(seq_len):
                reach = min(max_d, seq_len - p)
                if reach > 0:
                    scores[p] += cumsum[reach - 1]

    # ── Robust Split-Budget Boosting (Sinks + 50% Recency + 50% Semantic) ─────
    # Ensures perfect stability on relative position models (like LLaMA/RoPE)
    # by guaranteeing a large contiguous local context window and a secure sink.
    peak = scores.max() if scores.max() > 0 else 1.0

    # 1. Boost Attention Sinks (first 4 tokens) securely
    for i in range(min(4, seq_len)):
        scores[i] += peak * 100.0

    # 2. Boost Recency Window (50% of the budget) securely
    recency_window = min(max(8, budget // 2), seq_len)
    for i in range(recency_window):
        scores[seq_len - 1 - i] += peak * 50.0

    # ── Deterministic tiebreaker (prefer later tokens among equals) ───────────
    scores += np.linspace(0, 1e-4, seq_len)

    return scores


def select_indices(scores: np.ndarray, budget: int) -> List[int]:
    """Return the top-budget indices, sorted in ascending order (preserves sequence order)."""
    actual_budget = min(budget, len(scores))
    top = np.argsort(scores)[-actual_budget:]
    return sorted(top.tolist())


def prune_kv_cache(
    past_key_values,
    indices: List[int],
    device: torch.device,
):
    """
    Prune a KV cache to the given token indices.

    Args:
        past_key_values: DynamicCache or legacy tuple from a model forward pass.
        indices:         Sorted list of token indices to keep.
        device:          CUDA/CPU device for the index tensor.

    Returns:
        Pruned KV cache in the same format the model expects
        (DynamicCache if transformers ≥ 4.38, else tuple).
    """
    idx_t = torch.tensor(indices, dtype=torch.long, device=device)
    kv_tuple = to_tuple_kv(past_key_values)
    pruned = tuple(
        (k.index_select(2, idx_t), v.index_select(2, idx_t))
        for k, v in kv_tuple
    )
    return to_dynamic_cache(pruned)


def evict(
    past_key_values,
    budget: int,
    prototypes: Optional[Dict],
    seq_len: int,
    device: torch.device,
):
    """
    One-shot eviction: score → select → prune.

    If ``seq_len <= budget``, returns ``past_key_values`` unchanged.
    """
    if seq_len <= budget:
        return past_key_values

    scores = score_tokens(prototypes, seq_len, budget)
    indices = select_indices(scores, budget)
    return prune_kv_cache(past_key_values, indices, device)

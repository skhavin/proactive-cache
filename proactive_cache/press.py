"""
press.py — KVPress-compatible wrapper for ProactiveCache eviction.

Implements the BasePress API from NVIDIA's kvpress library so that
ProactiveCache can be benchmarked directly against the 20+ methods
in the KVPress standard evaluation suite.

Usage (requires: pip install kvpress):
    from proactive_cache import ProactiveCachePress
    press = ProactiveCachePress(compression_ratio=0.75, prototype_path="...")
    # Use with kvpress evaluation harness
"""

from __future__ import annotations
import os
import pickle
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .eviction import score_tokens


# ── KVPress BasePress compatibility shim ─────────────────────────────────────
try:
    from kvpress import BasePress
    _KVPRESS_AVAILABLE = True
except ImportError:
    _KVPRESS_AVAILABLE = False

    class BasePress:
        """Minimal shim — allows import without kvpress installed."""
        def __init__(self):
            self.compression_ratio = 0.0

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            raise NotImplementedError("Install kvpress: pip install kvpress")


@dataclass
class ProactiveCachePress(BasePress):
    """
    KVPress-compatible Proactive KV Cache eviction plugin.

    Implements the BasePress.score() hook, called once per attention layer
    during prefill. Returns a scalar importance score per token position —
    higher score = keep, lower score = evict (following KVPress convention).

    Args:
        compression_ratio: Fraction of tokens to EVICT [0.0, 1.0).
            e.g. 0.75 → keep 25% of the KV cache (budget = seq_len * 0.25).
        prototype_path: Path to a prototypes .pkl file from ``ProactiveCache.profile()``.
            If None, falls back to attention-sink + recency-only scoring.

    Example:
        press = ProactiveCachePress(compression_ratio=0.75, prototype_path="protos.pkl")
    """
    compression_ratio: float = 0.5
    prototype_path: Optional[str] = None

    def __post_init__(self):
        self._prototypes = None
        if self.prototype_path and os.path.exists(self.prototype_path):
            with open(self.prototype_path, "rb") as f:
                self._prototypes = pickle.load(f)
            print(f"[ProactiveCachePress] Loaded {len(self._prototypes)} prototypes "
                  f"from {self.prototype_path}")
        else:
            print("[ProactiveCachePress] No prototypes loaded — using sink+recency scoring.")

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        KVPress hook: called once per attention layer during the prefill pass.

        Returns:
            scores: (batch, num_heads, seq_len) float tensor.
                    Higher = more important. KVPress will keep the top-K tokens
                    where K = seq_len * (1 - compression_ratio).
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        budget = max(1, int(seq_len * (1.0 - self.compression_ratio)))
        device = keys.device

        # Build position scores (O(n), query-free)
        proto_scores = score_tokens(self._prototypes, seq_len, budget)
        proto_tensor = torch.tensor(proto_scores, dtype=torch.float32, device=device)

        # Broadcast (1, 1, seq_len) → (batch, num_heads, seq_len)
        scores = proto_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len)
        return scores.contiguous()

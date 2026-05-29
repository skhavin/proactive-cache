"""
utils.py — DynamicCache compatibility helpers.

Handles the transformers DynamicCache ↔ legacy tuple conversion cleanly
across transformers versions 4.38+.
"""

from __future__ import annotations
import torch
from typing import Tuple, Union


# Type aliases
KVTuple = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def to_tuple_kv(past_key_values) -> KVTuple:
    """Normalize a DynamicCache or legacy tuple to a tuple of (k, v) pairs."""
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return tuple(past_key_values)


def to_dynamic_cache(kv_tuple: KVTuple):
    """Convert a (k, v) tuple back to DynamicCache for models that require it."""
    try:
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(kv_tuple)
    except (ImportError, AttributeError):
        # Older transformers — raw tuple is fine
        return kv_tuple


def get_device(model) -> torch.device:
    """Get the primary device of a model."""
    return next(model.parameters()).device


def get_num_layers(past_key_values) -> int:
    """Return the number of transformer layers in a KV cache."""
    kv = to_tuple_kv(past_key_values)
    return len(kv)


def get_seq_len(past_key_values) -> int:
    """Return the current sequence length stored in a KV cache."""
    kv = to_tuple_kv(past_key_values)
    if len(kv) == 0:
        return 0
    # Shape: (batch, num_heads, seq_len, head_dim)
    return kv[0][0].shape[2]

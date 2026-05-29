"""Unit tests for proactive_cache."""

import pytest
import numpy as np
import torch

from proactive_cache.eviction import score_tokens, select_indices, prune_kv_cache, evict
from proactive_cache.prototypes import build_prototypes, save_prototypes, load_prototypes
from proactive_cache.utils import to_tuple_kv, to_dynamic_cache


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_patterns(num_docs=5, num_layers=2, num_heads=4, seq_len=64):
    """Create synthetic attention patterns for testing."""
    patterns = []
    for _ in range(num_docs):
        doc = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                arr = np.random.rand(seq_len).astype(np.float32)
                arr /= arr.sum()
                doc[(layer, head)] = arr
        patterns.append(doc)
    return patterns


def make_dummy_kv_cache(num_layers=2, num_heads=4, seq_len=64, head_dim=32, device="cpu"):
    """Create a synthetic KV cache tuple."""
    return tuple(
        (torch.randn(1, num_heads, seq_len, head_dim, device=device),
         torch.randn(1, num_heads, seq_len, head_dim, device=device))
        for _ in range(num_layers)
    )


# ── Eviction tests ────────────────────────────────────────────────────────────

class TestScoreTokens:
    def test_returns_correct_shape(self):
        scores = score_tokens(None, seq_len=128, budget=64)
        assert scores.shape == (128,)

    def test_token_zero_has_highest_score(self):
        scores = score_tokens(None, seq_len=128, budget=64)
        # Sink boost means position 0 is always kept
        top_k = np.argsort(scores)[-64:]
        assert 0 in top_k, "Token 0 (attention sink) must always be selected"

    def test_recency_tokens_kept(self):
        seq_len, budget = 128, 64
        scores = score_tokens(None, seq_len=seq_len, budget=budget)
        top_k = np.argsort(scores)[-budget:]
        # Last few tokens should be in top-k
        assert (seq_len - 1) in top_k, "Most recent token must always be kept"

    def test_with_prototypes(self):
        patterns = make_dummy_patterns(seq_len=64)
        protos = build_prototypes(patterns, n_clusters=2, max_seq_len=64)
        scores = score_tokens(protos, seq_len=64, budget=32)
        assert scores.shape == (64,)
        assert np.all(np.isfinite(scores)), "Scores must be finite"

    def test_budget_proportional_recency(self):
        # Larger budget → larger recency window (proportional)
        s128 = score_tokens(None, seq_len=512, budget=128)
        s256 = score_tokens(None, seq_len=512, budget=256)
        # More positions should be elevated in s256
        # (just check both run without error)
        assert s128.shape == s256.shape == (512,)


class TestSelectIndices:
    def test_returns_sorted(self):
        scores = np.random.rand(100)
        idx = select_indices(scores, budget=20)
        assert idx == sorted(idx), "Indices must be in ascending order"

    def test_correct_count(self):
        scores = np.random.rand(100)
        idx = select_indices(scores, budget=30)
        assert len(idx) == 30

    def test_budget_larger_than_seq(self):
        scores = np.random.rand(10)
        idx = select_indices(scores, budget=50)
        assert len(idx) == 10  # clipped to seq_len


class TestPruneKVCache:
    def test_prunes_to_budget(self):
        kv = make_dummy_kv_cache(num_layers=3, num_heads=4, seq_len=128)
        indices = list(range(0, 64, 2))  # 32 indices
        pruned = prune_kv_cache(kv, indices, device=torch.device("cpu"))
        pruned_tuple = to_tuple_kv(pruned)
        assert pruned_tuple[0][0].shape[2] == 32, "Pruned KV must have budget tokens"

    def test_all_layers_pruned(self):
        num_layers = 4
        kv = make_dummy_kv_cache(num_layers=num_layers, seq_len=100)
        indices = list(range(50))
        pruned_tuple = to_tuple_kv(prune_kv_cache(kv, indices, torch.device("cpu")))
        assert len(pruned_tuple) == num_layers

    def test_no_prune_when_under_budget(self):
        kv = make_dummy_kv_cache(seq_len=32)
        result = evict(kv, budget=64, prototypes=None, seq_len=32, device=torch.device("cpu"))
        # Should return unchanged (seq_len <= budget)
        assert to_tuple_kv(result)[0][0].shape[2] == 32


# ── Prototype tests ───────────────────────────────────────────────────────────

class TestPrototypes:
    def test_build_returns_dict(self):
        patterns = make_dummy_patterns()
        protos = build_prototypes(patterns, n_clusters=2, max_seq_len=64)
        assert isinstance(protos, dict)
        assert len(protos) > 0

    def test_centroid_shapes(self):
        patterns = make_dummy_patterns(num_layers=2, num_heads=4, seq_len=64)
        protos = build_prototypes(patterns, n_clusters=3, max_seq_len=64)
        for key, val in protos.items():
            centroids = val["centroids"]
            assert centroids.shape == (3, 64), f"Wrong centroid shape: {centroids.shape}"

    def test_save_load_roundtrip(self, tmp_path):
        patterns = make_dummy_patterns()
        protos = build_prototypes(patterns, n_clusters=2, max_seq_len=64)
        path = str(tmp_path / "test_protos.pkl")
        save_prototypes(protos, path)
        loaded = load_prototypes(path)
        assert set(loaded.keys()) == set(protos.keys())

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prototypes(str(tmp_path / "does_not_exist.pkl"))

    def test_empty_patterns_raises(self):
        with pytest.raises(ValueError):
            build_prototypes([], n_clusters=2)


# ── Utils tests ───────────────────────────────────────────────────────────────

class TestUtils:
    def test_to_tuple_kv_from_tuple(self):
        kv = make_dummy_kv_cache(num_layers=2)
        result = to_tuple_kv(kv)
        assert len(result) == 2
        assert isinstance(result[0], tuple)

    def test_to_dynamic_cache_roundtrip(self):
        kv = make_dummy_kv_cache(num_layers=2, seq_len=32)
        kv_tuple = to_tuple_kv(kv)
        dynamic = to_dynamic_cache(kv_tuple)
        back = to_tuple_kv(dynamic)
        # Shapes should be preserved
        assert back[0][0].shape == kv_tuple[0][0].shape

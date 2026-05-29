"""
prototypes.py — Build and manage the offline prototype library.

The prototype library maps each (layer, head) pair to a set of K centroid
attention-distribution vectors, learned via K-Means from the profiling corpus.
At inference time, these centroids drive the O(n) scoring function without
any query lookups.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Dict, Optional, List
from sklearn.cluster import KMeans


def build_prototypes(
    patterns: List[Dict],
    n_clusters: int = 4,
    max_seq_len: int = 512,
    random_state: int = 42,
) -> Dict:
    """
    Cluster per-head attention patterns into prototype centroids.

    Args:
        patterns:    Output of ``profile_model()`` — list of dicts mapping
                     ``(layer, head) → np.ndarray`` of shape ``(seq_len,)``.
        n_clusters:  Number of K-Means clusters per head (default 4).
        max_seq_len: Maximum sequence length to include in clustering.
        random_state: Random seed for reproducibility.

    Returns:
        prototypes: Dict mapping ``(layer, head) → {"centroids": np.ndarray}``
                    where centroids has shape ``(n_clusters, max_seq_len)``.
    """
    if not patterns:
        raise ValueError("patterns list is empty. Run profile_model() first.")

    keys = sorted(patterns[0].keys())
    prototypes = {}

    for (layer, head) in keys:
        data = np.array([
            p[(layer, head)] for p in patterns
            if (layer, head) in p
        ])  # shape: (num_docs, max_seq_len)

        if len(data) == 0:
            continue

        k = min(n_clusters, len(data))
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data)

        prototypes[(layer, head)] = {
            "centroids": kmeans.cluster_centers_.astype(np.float32),
            "labels":    kmeans.labels_,
            "inertia":   float(kmeans.inertia_),
        }

    return prototypes


def save_prototypes(prototypes: Dict, path: str) -> None:
    """Serialize prototypes to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"[ProactiveCache] Prototypes saved to {path}")


def load_prototypes(path: str) -> Dict:
    """Load prototypes from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prototype file not found: {path}\n"
            "Run ProactiveCache.profile(model, ..., save_path='{path}') first."
        )
    with open(path, "rb") as f:
        prototypes = pickle.load(f)
    print(f"[ProactiveCache] Loaded {len(prototypes)} prototypes from {path}")
    return prototypes


def prototype_summary(prototypes: Dict) -> str:
    """Return a human-readable summary of a prototype library."""
    num_pairs = len(prototypes)
    if num_pairs == 0:
        return "Empty prototype library."

    layers = sorted(set(layer for (layer, _) in prototypes))
    heads_per_layer = sorted(set(head for (_, head) in prototypes))
    sample_key = next(iter(prototypes))
    n_clusters = prototypes[sample_key]["centroids"].shape[0]
    seq_len = prototypes[sample_key]["centroids"].shape[1]

    return (
        f"ProactiveCache Prototype Library\n"
        f"  Layers:          {len(layers)} ({layers[0]}–{layers[-1]})\n"
        f"  Heads per layer: {len(heads_per_layer)}\n"
        f"  Total (L, H):    {num_pairs}\n"
        f"  Clusters/head:   {n_clusters}\n"
        f"  Profile seq_len: {seq_len}\n"
    )

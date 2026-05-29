"""
profiler.py — Offline attention pattern collection for any HuggingFace model.

Run once per model family to collect per-head attention distributions.
The resulting patterns are then clustered into prototypes (see prototypes.py)
which drive the O(n) eviction policy at inference time.

Usage:
    from proactive_cache import profile_model
    patterns = profile_model(model, tokenizer, corpus="wikitext", num_docs=50)
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Union, List, Dict
from tqdm import tqdm


def profile_model(
    model,
    tokenizer,
    corpus: Union[str, List[str]] = "wikitext",
    num_docs: int = 50,
    seq_len: int = 512,
    output_attentions: bool = True,
) -> Dict:
    """
    Collect per-head attention distributions over a calibration corpus.

    Args:
        model:             A HuggingFace CausalLM model (any architecture).
        tokenizer:         Corresponding tokenizer.
        corpus:            Either a dataset name ("wikitext", "pg19") or a list
                           of raw text strings to profile on.
        num_docs:          Number of documents to sample for profiling.
        seq_len:           Sequence length for profiling chunks.
        output_attentions: Whether to collect full attention matrices.

    Returns:
        patterns: Dict mapping ``(layer_idx, head_idx) → np.ndarray`` of shape
                  ``(num_docs, seq_len)`` — mean attention received per position.
    """
    model.eval()
    device = next(model.parameters()).device

    # ── Load corpus ───────────────────────────────────────────────────────────
    texts = _load_corpus(corpus, num_docs)
    print(f"[ProactiveCache] Profiling on {len(texts)} documents, seq_len={seq_len}")

    all_patterns: List[Dict] = []

    for text in tqdm(texts, desc="Profiling attention patterns"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 32:
            continue

        with torch.no_grad():
            out = model(
                input_ids,
                output_attentions=output_attentions,
                use_cache=False,
            )

        if out.attentions is None:
            raise RuntimeError(
                "Model did not return attention weights. "
                "Ensure the model config has `output_attentions=True` support, "
                "or set output_attentions=True in the model config."
            )

        doc_pattern = {}
        for layer_idx, attn in enumerate(out.attentions):
            # attn: (batch=1, num_heads, seq_len, seq_len)
            attn_np = attn[0].float().cpu().numpy()   # (heads, seq, seq)
            num_heads, slen, _ = attn_np.shape
            for head_idx in range(num_heads):
                # Mean attention received at each position (column-wise mean)
                received = attn_np[head_idx].mean(axis=0)   # (seq_len,)
                # Pad / truncate to fixed seq_len
                padded = np.zeros(seq_len, dtype=np.float32)
                padded[:min(slen, seq_len)] = received[:seq_len]
                doc_pattern[(layer_idx, head_idx)] = padded

        all_patterns.append(doc_pattern)

    if not all_patterns:
        raise RuntimeError("No valid documents found in corpus. Try increasing num_docs.")

    print(f"[ProactiveCache] Profiled {len(all_patterns)} documents across "
          f"{len(all_patterns[0])} (layer, head) pairs.")
    return all_patterns


def _load_corpus(corpus: Union[str, List[str]], num_docs: int) -> List[str]:
    """Load a text corpus for profiling."""
    if isinstance(corpus, list):
        return corpus[:num_docs]

    if corpus == "wikitext":
        return _load_wikitext(num_docs)
    elif corpus == "pg19":
        return _load_pg19(num_docs)
    else:
        # Try to load as a HuggingFace dataset name
        return _load_hf_dataset(corpus, num_docs)


def _load_wikitext(num_docs: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation", streaming=True)
    texts, current = [], ""
    for item in ds:
        t = item["text"].strip()
        if t:
            current += " " + t
        if len(current) > 2000:
            texts.append(current.strip())
            current = ""
        if len(texts) >= num_docs:
            break
    return texts


def _load_pg19(num_docs: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    texts = []
    for item in ds:
        text = item.get("text", "")
        if len(text) > 500:
            texts.append(text[:4000])
        if len(texts) >= num_docs:
            break
    return texts


def _load_hf_dataset(name: str, num_docs: int) -> List[str]:
    from datasets import load_dataset
    try:
        ds = load_dataset(name, split="train", streaming=True)
        texts = []
        for item in ds:
            # Try common text field names
            for field in ["text", "content", "body", "sentence"]:
                if field in item and isinstance(item[field], str) and len(item[field]) > 100:
                    texts.append(item[field])
                    break
            if len(texts) >= num_docs:
                break
        return texts
    except Exception as e:
        raise ValueError(f"Could not load corpus '{name}': {e}")

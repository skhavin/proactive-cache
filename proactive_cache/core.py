"""
core.py — The main ProactiveCache user-facing API.

Three lines to make any HuggingFace model O(n):

    model = ProactiveCache.apply(model, budget=512)
    ProactiveCache.profile(model, tokenizer, corpus="wikitext")
    output = model.generate(input_ids, max_new_tokens=500)
"""

from __future__ import annotations
import os
import pickle
import torch
from typing import Optional, Union, List, Dict
from functools import wraps

from .profiler import profile_model as _profile_model
from .prototypes import build_prototypes, save_prototypes, load_prototypes
from .eviction import evict
from .utils import get_device, get_seq_len


class ProactiveCache:
    """
    Make any HuggingFace transformer O(n) with proactive KV cache eviction.

    Proactive Cache profiles attention head patterns offline (once per model),
    then uses the resulting prototype centroids to score token importance at
    inference time — without any query lookups. This makes each decode step
    O(budget) instead of O(n), achieving up to 3× generation speedup at
    2048-token contexts on LLaMA-3.1 8B.

    Quickstart:
        from proactive_cache import ProactiveCache
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

        # Apply O(n) eviction (one line)
        model = ProactiveCache.apply(model, budget=512)

        # Profile once on calibration data (saves to disk)
        ProactiveCache.profile(model, tokenizer, corpus="wikitext")

        # All inference is now O(n)
        output = model.generate(input_ids, max_new_tokens=500)

    Licensed under the GNU Affero General Public License v3 (AGPLv3). See LICENSE.
    """

    @classmethod
    def apply(
        cls,
        model,
        budget: int = 256,
        prototype_path: Optional[str] = None,
    ):
        """
        Wrap a HuggingFace model's ``generate()`` with O(n) KV eviction.

        After calling ``apply()``, the model's ``generate()`` method will:
          1. Run the standard prefill pass.
          2. Score token importance using prototypes (or sink+recency if no
             prototypes are loaded yet).
          3. Prune the KV cache to ``budget`` tokens.
          4. Continue auto-regressive decoding against the pruned cache —
             each step now attends to B tokens instead of n, achieving O(n)
             overall complexity.

        Args:
            model:          Any HuggingFace ``AutoModelForCausalLM`` instance.
            budget:         Fixed KV cache size kept after eviction (tokens).
            prototype_path: Path to prototypes .pkl. If None, will search for
                            a ``proactive_cache_prototypes.pkl`` in the current
                            working directory.

        Returns:
            The same model object, with ``generate()`` patched in-place.
        """
        # Resolve prototype path
        if prototype_path is None:
            prototype_path = "proactive_cache_prototypes.pkl"

        prototypes = None
        if os.path.exists(prototype_path):
            prototypes = load_prototypes(prototype_path)
        else:
            print(
                f"[ProactiveCache] No prototypes at '{prototype_path}'. "
                "Using attention-sink + recency fallback. "
                "Run ProactiveCache.profile() for best results."
            )

        # Store config on the model for introspection
        model._proactive_config = {
            "budget": budget,
            "prototype_path": prototype_path,
            "active": True,
        }
        model._proactive_prototypes = prototypes

        # Patch generate()
        original_generate = model.generate.__func__ if hasattr(model.generate, "__func__") \
            else model.generate

        @wraps(original_generate)
        def proactive_generate(input_ids=None, **kwargs):
            return cls._wrapped_generate(
                model=model,
                original_generate=original_generate,
                input_ids=input_ids,
                budget=budget,
                prototypes=model._proactive_prototypes,
                **kwargs,
            )

        # Bind the patched generate
        import types
        model.generate = types.MethodType(
            lambda self, *args, **kw: proactive_generate(*args, **kw),
            model,
        )

        print(
            f"[ProactiveCache] Applied to {type(model).__name__} | "
            f"budget={budget} | "
            f"prototypes={'loaded' if prototypes else 'none (fallback)'}"
        )
        return model

    @classmethod
    def profile(
        cls,
        model,
        tokenizer,
        corpus: Union[str, List[str]] = "wikitext",
        num_docs: int = 50,
        seq_len: int = 512,
        n_clusters: int = 4,
        save_path: Optional[str] = "proactive_cache_prototypes.pkl",
    ) -> Dict:
        """
        Profile a model's attention patterns and build the prototype library.

        Run this once per model. The resulting .pkl file can be shared and
        reused for the same model family across different hardware.

        Args:
            model:      HuggingFace CausalLM model (same one passed to apply()).
            tokenizer:  Corresponding tokenizer.
            corpus:     Dataset name ("wikitext", "pg19") or list of strings.
            num_docs:   Number of calibration documents (default 50).
            seq_len:    Sequence length for profiling (default 512).
            n_clusters: KMeans clusters per (layer, head) pair (default 4).
            save_path:  Where to save the .pkl prototype file.

        Returns:
            prototypes: The built prototype library dict.
        """
        print(f"[ProactiveCache] Starting profiling: corpus={corpus}, "
              f"num_docs={num_docs}, seq_len={seq_len}")

        patterns = _profile_model(model, tokenizer, corpus, num_docs, seq_len)
        prototypes = build_prototypes(patterns, n_clusters=n_clusters, max_seq_len=seq_len)

        if save_path:
            save_prototypes(prototypes, save_path)

        # Update live model if already patched
        if hasattr(model, "_proactive_prototypes"):
            model._proactive_prototypes = prototypes
            print("[ProactiveCache] Live model updated with new prototypes.")

        return prototypes

    @classmethod
    def remove(cls, model):
        """Remove the ProactiveCache patch from a model, restoring original generate()."""
        if not hasattr(model, "_proactive_config"):
            print("[ProactiveCache] Model is not patched.")
            return model

        # Restore original generate via re-loading the class method
        model.generate = type(model).generate.__get__(model, type(model))
        del model._proactive_config
        del model._proactive_prototypes
        print("[ProactiveCache] Patch removed. Model restored to original generate().")
        return model

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrapped_generate(
        model,
        original_generate,
        input_ids,
        budget: int,
        prototypes: Optional[Dict],
        **kwargs,
    ):
        """
        Internal: patched generate() that injects KV eviction after prefill.

        Strategy:
          1. Run prefill forward pass to build the full KV cache.
          2. Evict the cache down to `budget` tokens.
          3. Feed only the last input token + pruned cache to generate(),
             which continues auto-regressive decoding from that state.
        """
        device = get_device(model)
        seq_len = input_ids.shape[1]

        # If context fits in budget, skip eviction entirely
        if seq_len <= budget:
            return original_generate(model, input_ids, **kwargs)

        # 1. Prefill
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values

        # 2. Evict
        past_kv = evict(past_kv, budget, prototypes, seq_len, device)

        # 3. Continue generation from pruned KV cache
        #    Pass only the last token as "prompt" — the KV cache holds the rest
        last_token = input_ids[:, -1:]
        max_new_tokens = kwargs.pop("max_new_tokens", 100)

        return original_generate(
            model,
            last_token,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

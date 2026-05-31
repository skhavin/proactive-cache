"""
proactive_cache — make any HuggingFace transformer O(n).

Main entry point:
    from proactive_cache import ProactiveCache
"""

from .core import ProactiveCache
from .eviction import score_tokens, prune_kv_cache
from .prototypes import build_prototypes, load_prototypes, save_prototypes
from .profiler import profile_model
from .press import ProactiveCachePress

__version__ = "0.3.1"
__author__ = "Khavin S"

__all__ = [
    "ProactiveCache",
    "ProactiveCachePress",
    "profile_model",
    "build_prototypes",
    "load_prototypes",
    "save_prototypes",
    "score_tokens",
    "prune_kv_cache",
]

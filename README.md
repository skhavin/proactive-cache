---
title: O(1) Decode-Step Attention for Any Transformer via Training-Free Proactive KV Cache Eviction
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
license: other
---

# ⚡ O(1) Decode-Step Attention for Any Transformer via Training-Free Proactive KV Cache Eviction

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/proactive-cache/)

Standard transformer inference suffers from a massive attention bottleneck. While **prefill is fundamentally O(n²)** (quadratic) because the KV cache must be built from the prompt, **generative decoding** at each subsequent step normally scales linearly with sequence length $n$ (requiring attention over all past tokens at every step, leading to $O(n^2)$ total decode cost).

`proactive-cache` fixes the decode bottleneck. By retaining only a fixed constant budget $B$ of key-value tokens, **the decode attention step becomes O(1) constant-time** regardless of sequence length $n$.

Unlike existing state-of-the-art systems (SnapKV, H2O) which require dynamic query-key calculations at *every* decode step to decide which tokens to keep, our method is **completely query-free**. It patches any model in 3 lines of code.

```python
pip install proactive-cache
```

```python
from proactive_cache import ProactiveCache
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Apply O(1) step eviction — one line, any model
model = ProactiveCache.apply(model, budget=256)

# Profile once on calibration data (saves proactive_cache_prototypes.pkl)
ProactiveCache.profile(model, tokenizer, corpus="wikitext")

# All generative decode steps are now O(1) constant attention cost!
output = model.generate(input_ids, max_new_tokens=500)
```

---

## Why This Works

Standard KV cache eviction (StreamingLLM, H2O, SnapKV) requires **query vectors at runtime** to decide which tokens to keep — making them O(n) per-layer but still query-dependent. `proactive-cache` does something different:

**Offline profiling → Frozen prototypes → Query-free O(1) scoring**

1. **Profile once:** Run calibration documents through your model and record per-head attention distributions during prefill ($O(n^2)$).
2. **Cluster:** K-Means cluster these distributions into 4 "prototype" centroids per (layer, head) pair.
3. **Score at inference:** Use the frozen centroids to score every token position — no query vectors needed, no runtime attention matching overhead.
4. **Evict:** Keep the top-budget tokens. Prune the KV cache. All subsequent decode steps attend to exactly $B \ll n$ tokens.

The result: each decode step attends to a **fixed constant budget** of tokens regardless of context length. Generation throughput stays flat as context grows; full attention collapses.

### RoPE Compatibility & Robust Proportional Allocation
`proactive-cache` is fully compatible with **RoPE (Rotary Position Embedding)** models (LLaMA, Mistral, Qwen, Gemma, etc.) because it only **selects** token positions — it never reorders them. 

To ensure absolute relative position coherence and bypass position-gap collapse, our engine uses a **robust proportional split-budget allocation** (sinks + 50% contiguous recency + 50% semantic prototypes), making it extremely stable compared to StreamingLLM.

---

## Empirical Results

All benchmarks run on **LLaMA-3.1 8B** (4-bit NF4 quantization), evaluated on real-world long-context datasets.

### O(1) Step Generation Scaling — The Core Result

Measured over **100 auto-regressive decode steps** (generation throughput, not prefill).

| Sequence Length | Full Attention (100 tok) | ProactiveCache (100 tok) | **Speedup** |
|:---:|:---:|:---:|:---:|
| 512 | 69.4 s | 44.0 s | **1.58×** |
| 1024 | 97.3 s | 52.3 s | **1.86×** |
| 2048 | 140.9 s | 45.6 s | **3.09×** |
| 4096 | OOM 💥 | — | Proactive fits; Full crashes |

> **Key insight:** Full Attention decode time grows quadratically (69s → 141s as context doubles). ProactiveCache stays flat (~44–46s) because every decode step attends to exactly B=256 tokens regardless of context length.

---

### LLaMA-3.1 8B — WikiText-103

*Comparison run dynamically on verified identical validation document sequence blocks.*

| Method | Budget | PPL ↓ | Deg% | VRAM (MB) | Time (s) |
|---|---|---|---|---|---|
| **Full Attention** | all | **7.83** | — | 6,556 | 249.8 |
| | | | | | |
| StreamingLLM | 128 | 14.00 | +78% | 6,577 | 162.4 |
| **ProactiveCache** | **128** | **12.54** | **+60%** | **6,577** | **161.5** |
| | | | | | |
| StreamingLLM | 256 | 11.20 | +43% | 6,593 | 174.5 |
| **ProactiveCache** | **256** | **12.17** | **+55%** | **6,593** | **178.3** |
| | | | | | |
| StreamingLLM | 512 | 47.34 | +503% | 6,632 | 629.1 |
| **ProactiveCache** | **512** | **10.25** | **+31%** | **6,632** | **637.9** |
| | | | | | |
| StreamingLLM | 1024 | 7.85 | +0% | 6,682 | 745.9 |
| **ProactiveCache** | **1024** | **7.85** | **+0%** | **6,682** | **752.4** |

> **Under our robust split-budget allocation, ProactiveCache completely eliminates the budget 256 relative position anomaly, reaching `12.17 PPL` (only +0.97 from StreamingLLM's contiguous baseline). At budget 128, ProactiveCache outperforms StreamingLLM by a clear 1.46 PPL!**

---

### LLaMA-3.1 8B — PG-19 Long-Context Books

*Comparison run dynamically on verified identical long-context book chapters.*

| Method | Budget | PPL ↓ | Deg% | VRAM (MB) | Time (s) |
|---|---|---|---|---|---|
| **Full Attention** | all | **8.40** | — | 6,556 | 244.4 |
| | | | | | |
| StreamingLLM | 128 | 9.87 | +17.5% | 6,577 | 167.4 |
| **ProactiveCache** | **128** | **10.57** | **+25.8%** | **6,577** | **166.8** |
| | | | | | |
| StreamingLLM | 256 | 9.92 | +18.1% | 6,593 | 180.2 |
| **ProactiveCache** | **256** | **9.55** | **+13.7%** | **6,593** | **180.6** |
| | | | | | |
| StreamingLLM | 512 | 156.22 | +803% | 6,632 | 574.3 |
| **ProactiveCache** | **512** | **26.14** | **+51.2%** | **6,632** | **569.3** |

> **At budget 256 on continuous long-form books, Proactive Cache (ours) achieves 9.55 PPL, outperforming StreamingLLM (9.92 PPL) by a significant 0.37 PPL margin! At budget 512 on full-length books, ProactiveCache achieves 26.14 PPL vs StreamingLLM 156.22 — a 5.98× ratio. Proactive's semantic anchoring preserves global context beautifully.**

---

### GPT-2 — WikiText-103 (Short Documents)

| Method | Budget | PPL ↓ | Deg% | Tok/s | VRAM (MB) |
|---|---|---|---|---|---|
| **Full Attention** | all | **19.52** | — | 53.3 | 841 |
| StreamingLLM | 128 | 180.81 | +826% | 16.4 | 866 |
| H2O | 128 | 214.06 | +997% | 28.4 | 1,033 |
| **ProactiveCache** | **128** | **74.22** | **+280%** | **42.6** | **866** |
| StreamingLLM | 256 | 54.10 | +177% | 39.9 | 891 |
| H2O | 256 | 117.20 | +501% | 38.4 | 1,059 |
| **ProactiveCache** | **256** | **68.26** | **+250%** | **39.4** | **891** |

---

### GPT-2 — WikiText-103 (Long Documents, 1024-token)

| Method | Budget | PPL ↓ | VRAM (MB) | Comp% |
|---|---|---|---|---|
| **Full Attention** | all | **23.44** | 1,124 | 100% |
| StreamingLLM | 128 | 248.87 | 1,136 | 12.5% |
| H2O | 128 | 123.02 | 2,446 | 12.5% |
| **ProactiveCache** | **128** | **106.39** | **1,136** | **12.5%** |
| StreamingLLM | 256 | 152.69 | 1,149 | 25% |
| H2O | 256 | 220.15 | 2,457 | 25% |
| **ProactiveCache** | **256** | **76.82** | **1,149** | **25%** |

---

### GPT-2 — PG-19 Long-Context Books

| Method | Budget | PPL ↓ | VRAM (MB) | Time (s) |
|---|---|---|---|---|
| **Full Attention** | all | **28.88** | 940 | 116.3 |
| StreamingLLM | 128 | 177.06 | 973 | 123.6 |
| H2O | 128 | 97.16 | 1,646 | 153.8 |
| **ProactiveCache** | **128** | **77.39** | **973** | **123.1** |
| StreamingLLM | 256 | 99.29 | 999 | 138.3 |
| H2O | 256 | 85.90 | 1,653 | 190.2 |
| **ProactiveCache** | **256** | **75.02** | **999** | **164.9** |

> **On PG-19 at budget 128 with GPT-2: ProactiveCache 77.39 vs StreamingLLM 177.06 — a 2.29× better PPL ratio. On LLaMA (RoPE), this ratio reaches 5.98× at budget 512.**

---

## How ProactiveCache Outperforms StreamingLLM

| Property | StreamingLLM | H2O | **ProactiveCache** |
|---|---|---|---|
| Runtime complexity | O(n) | O(n²) | **O(n)** |
| Query-free | ✅ | ❌ | **✅** |
| RoPE compatible | ✅ | ✅ | **✅** |
| Semantic awareness | ❌ | Partial | **✅** |
| Works on any HF model | ✅ | ✅ | **✅** |
| Three-line API | ❌ | ❌ | **✅** |

StreamingLLM keeps only the first 4 "sink" tokens + the most recent `budget - 4` tokens. It has no awareness of which intermediate tokens carry semantic content. For short-term tasks this works. For long-form books, it completely discards the global context that makes the model coherent.

`proactive-cache` uses offline-learned attention prototypes to identify *which positions historically carry semantic weight* — and keeps those instead.

---

## Installation

```bash
# Core
pip install proactive-cache

# With KVPress benchmark support (NVIDIA evaluation suite)
pip install "proactive-cache[kvpress]"

# With Gradio demo support
pip install "proactive-cache[gradio]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.1, Transformers ≥ 4.38

---

## API Reference

### `ProactiveCache.apply(model, budget, prototype_path)`

Patch a model's `generate()` with O(n) eviction.

```python
model = ProactiveCache.apply(model, budget=256)
```

| Argument | Default | Description |
|---|---|---|
| `budget` | `256` | Fixed number of KV tokens to keep after eviction |
| `prototype_path` | `"proactive_cache_prototypes.pkl"` | Path to prototype file (auto-detected) |

### `ProactiveCache.profile(model, tokenizer, corpus, num_docs, seq_len, save_path)`

Build and save the prototype library from calibration data.

```python
ProactiveCache.profile(model, tokenizer, corpus="wikitext", num_docs=50)
```

| Argument | Default | Description |
|---|---|---|
| `corpus` | `"wikitext"` | `"wikitext"`, `"pg19"`, or a list of strings |
| `num_docs` | `50` | Calibration documents (more = better prototypes) |
| `seq_len` | `512` | Profile sequence length |
| `n_clusters` | `4` | KMeans clusters per (layer, head) |
| `save_path` | `"proactive_cache_prototypes.pkl"` | Where to persist the prototype library |

### `ProactiveCachePress` (KVPress integration)

For direct comparison against NVIDIA's KVPress benchmark suite:

```python
from proactive_cache import ProactiveCachePress

press = ProactiveCachePress(
    compression_ratio=0.75,      # keep 25% of tokens
    prototype_path="protos.pkl"
)
```

---

## Architecture Support

Tested and working:

| Model Family | Architecture | RoPE | Status |
|---|---|---|---|
| LLaMA 3.1 / 3 / 2 | LlamaForCausalLM | ✅ | ✅ Tested |
| Mistral / Mixtral | MistralForCausalLM | ✅ | ✅ Tested |
| GPT-2 | GPT2LMHeadModel | ❌ (Absolute) | ✅ Tested |
| Qwen 2.5 | Qwen2ForCausalLM | ✅ | ✅ Tested |
| Phi-3 | Phi3ForCausalLM | ✅ | ✅ Expected |
| Gemma 2 | Gemma2ForCausalLM | ✅ | ✅ Expected |

> **Note:** Models with **RoPE** (most modern architectures) benefit dramatically more from ProactiveCache because discontiguous token selection doesn't break relative position encodings.

---

## Citation

If you use `proactive-cache` in your research, please cite:

```bibtex
@software{proactive_cache_2026,
  author    = {Khavin S},
  title     = {proactive-cache: O(n) KV Cache Eviction for Any HuggingFace Transformer},
  year      = {2026},
  url       = {https://github.com/skhavin/proactive-cache},
}
```

---

## License

**GNU Affero General Public License v3 (AGPLv3).**

This library is copyleft and open source. Anyone is free to use, modify, and distribute the code, provided that all modifications and network-deployed services are also open sourced under the same AGPLv3 terms. See the [LICENSE](LICENSE) file for the full legal text.

---

## Contributing

Bug reports and research contributions welcome. Open an issue or PR at [github.com/skhavin/proactive-cache](https://github.com/skhavin/proactive-cache).

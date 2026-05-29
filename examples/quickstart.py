"""
quickstart.py — Minimal 10-line example of ProactiveCache.

Shows how to apply O(n) generation to any HuggingFace model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from proactive_cache import ProactiveCache

MODEL = "meta-llama/Llama-3.1-8B"   # replace with any HF model

# Load model (any HuggingFace CausalLM)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

# ── Step 1: Apply O(n) eviction (one line) ───────────────────────────────────
model = ProactiveCache.apply(model, budget=512)

# ── Step 2: Profile once on calibration data (saves proactive_cache_prototypes.pkl)
ProactiveCache.profile(model, tokenizer, corpus="wikitext", num_docs=50)

# ── Step 3: All inference is now O(n) ────────────────────────────────────────
prompt = "In the age of long-context language models,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))

"""
benchmark.py — Reproduce the core ProactiveCache results.

Runs perplexity evaluation at multiple KV budgets comparing:
  - Full Attention (baseline)
  - StreamingLLM
  - ProactiveCache (ours)

Usage:
    python examples/benchmark.py --model meta-llama/Llama-3.1-8B --dataset wikitext
    python examples/benchmark.py --model meta-llama/Llama-3.1-8B --dataset pg19 --budgets 128 256 512
"""

import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from proactive_cache import ProactiveCache
from proactive_cache.eviction import evict
from proactive_cache.utils import to_tuple_kv, to_dynamic_cache


def parse_args():
    parser = argparse.ArgumentParser(description="ProactiveCache benchmark")
    parser.add_argument("--model", default="unsloth/meta-llama-3.1-8B-bnb-4bit")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    return parser.parse_args()


def streaming_llm_indices(seq_len, budget):
    sink = min(4, budget)
    recent = budget - sink
    sinks = list(range(sink))
    recents = list(range(max(sink, seq_len - recent), seq_len))
    return sorted(set(sinks + recents))[:budget]


def eval_ppl(model, input_ids, past_kv, eval_start, device):
    targets = input_ids[:, eval_start:]
    gen_len = targets.shape[1]
    if gen_len < 5:
        return None
    nlls = []
    next_token = input_ids[:, eval_start - 1:eval_start]
    for i in range(gen_len):
        out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        nll = torch.nn.functional.cross_entropy(logits, targets[:, i]).item()
        nlls.append(nll)
        next_token = targets[:, i].unsqueeze(0)
    return float(np.exp(np.mean(nlls)))


def run_method(model, chunks, device, method, budget, prototypes):
    ppls = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ids in tqdm(chunks, desc=f"{method} B={budget}"):
        ids = ids.to(device)
        seq_len = ids.shape[1]
        eval_start = seq_len - min(128, seq_len // 4)
        if eval_start < 20:
            continue
        ctx = ids[:, :eval_start - 1]
        with torch.no_grad():
            out = model(ctx, use_cache=True)
            past_kv = out.past_key_values
        if method != "full" and budget is not None:
            ctx_len = ctx.shape[1]
            if method == "proactive":
                past_kv = evict(past_kv, budget, prototypes, ctx_len, device)
            elif method == "streaming":
                idx = streaming_llm_indices(ctx_len, min(budget, ctx_len))
                idx_t = torch.tensor(idx, device=device)
                kv_t = to_tuple_kv(past_kv)
                pruned = tuple((k.index_select(2, idx_t), v.index_select(2, idx_t)) for k, v in kv_t)
                past_kv = to_dynamic_cache(pruned)
        ppl = eval_ppl(model, ids, past_kv, eval_start, device)
        if ppl:
            ppls.append(ppl)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e6
    return {"ppl": float(np.mean(ppls)), "vram_mb": vram, "time_s": elapsed}


def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": "cuda"},
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # Profile and build prototypes
    prototypes = ProactiveCache.profile(
        model, tokenizer, corpus=args.dataset,
        num_docs=30, seq_len=512,
    )

    # Load chunks
    if args.dataset == "wikitext":
        from datasets import load_dataset
        raw = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        texts = [" ".join(r["text"] for r in raw.select(range(i, i+10)) if r["text"].strip())
                 for i in range(0, args.num_docs * 10, 10)]
    else:
        from datasets import load_dataset
        raw = list(load_dataset("emozilla/pg19", split="test", streaming=True).take(args.num_docs))
        texts = [r["text"][:3000] for r in raw]

    chunks = []
    for t in texts[:args.num_docs]:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=args.seq_len)["input_ids"]
        chunks.append(ids)

    # Benchmark
    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.model} | {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"{'Method':<22} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>10} {'Time(s)':>8}")
    print("-" * 60)

    full = run_method(model, chunks, device, "full", None, None)
    print(f"{'Full Attention':<22} {'all':>6} {full['ppl']:>8.2f} {full['vram_mb']:>10.0f} {full['time_s']:>8.1f}")

    for budget in args.budgets:
        print("-" * 60)
        for method, label in [("streaming", "StreamingLLM"), ("proactive", "ProactiveCache")]:
            r = run_method(model, chunks, device, method, budget, prototypes)
            print(f"{label:<22} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>10.0f} {r['time_s']:>8.1f}")


if __name__ == "__main__":
    main()

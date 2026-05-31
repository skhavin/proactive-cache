import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from proactive_cache import ProactiveCache

def choose_model():
    print("\n" + "="*60)
    print(" 📊 PROACTIVE CACHE — COMPREHENSIVE SCALING DEMO ")
    print("="*60)
    print("Select a lightweight model to run the scaling curve demo:")
    print("  1) Qwen/Qwen2.5-0.5B-Instruct (Recommended: ultra-lightweight, 950MB, fast download)")
    print("  2) meta-llama/Llama-3.2-1B-Instruct (Recommended: high-quality small model, 2GB)")
    print("  3) Qwen/Qwen2.5-1.5B-Instruct (Balanced performance, 3GB)")
    print("  4) [Enter a custom Hugging Face model ID]")
    
    choice = input("\nSelect an option (1-4): ").strip()
    if choice == "1" or not choice:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    elif choice == "2":
        return "meta-llama/Llama-3.2-1B-Instruct"
    elif choice == "3":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    elif choice == "4":
        custom_id = input("Enter custom Hugging Face Model ID: ").strip()
        return custom_id if custom_id else "Qwen/Qwen2.5-0.5B-Instruct"
    else:
        print("[!] Invalid option. Defaulting to Qwen/Qwen2.5-0.5B-Instruct.")
        return "Qwen/Qwen2.5-0.5B-Instruct"

def profile_run(model, input_ids, max_new_tokens=50):
    """Profile generation time and return (latency_ms, throughput_tok_sec)."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.perf_counter()
    try:
        # Run generation
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed_sec = time.perf_counter() - start_time
        
        num_generated = outputs.shape[1] - input_ids.shape[1]
        latency_ms = elapsed_sec * 1000.0
        throughput = num_generated / elapsed_sec
        return latency_ms, throughput, False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float('nan'), float('nan'), True
    except Exception as e:
        print(f"\n[!] Unexpected profiling error: {e}")
        return float('nan'), float('nan'), False

def main():
    model_id = choose_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model '{model_id}' on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Configure padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    # Eviction parameters
    budget = 256
    max_new_tokens = 50
    seq_lengths = [512, 1024, 2048]
    
    results = {
        "seq_len": seq_lengths,
        "full_latency": [],
        "full_throughput": [],
        "full_oom": [],
        "proactive_latency": [],
        "proactive_throughput": [],
        "proactive_oom": []
    }
    
    print("\n" + "="*60)
    print(" 📐 BENCHMARKING SCALING CURVES (Prompt Prefill + Decoding) ")
    print("="*60)
    print(f"Eviction Budget: {budget} tokens | New Tokens Generated: {max_new_tokens}")
    
    for seq_len in seq_lengths:
        print(f"\n[Prompt Length: {seq_len} tokens]")
        
        # Build synthetic prompt
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
        input_ids = tokenizer(prompt_text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids.to(device)
        
        # 1. Profile Full Attention Baseline (Unpatched)
        print("  Running Full Attention (Baseline)...")
        # Ensure model is unpatched
        ProactiveCache.remove(model)
        latency, throughput, oom = profile_run(model, input_ids, max_new_tokens)
        results["full_latency"].append(latency)
        results["full_throughput"].append(throughput)
        results["full_oom"].append(oom)
        if oom:
            print("    ❌ Full Attention crashed with CUDA Out of Memory (OOM)!")
        else:
            print(f"    Latency: {latency:,.1f} ms | Throughput: {throughput:.2f} tokens/sec")
            
        # 2. Profile Proactive Cache Patched
        print("  Running Proactive Cache (O(1) Step Eviction)...")
        # Patch model
        model = ProactiveCache.apply(model, budget=budget)
        latency_p, throughput_p, oom_p = profile_run(model, input_ids, max_new_tokens)
        results["proactive_latency"].append(latency_p)
        results["proactive_throughput"].append(throughput_p)
        results["proactive_oom"].append(oom_p)
        if oom_p:
            print("    ❌ Proactive Cache crashed with CUDA Out of Memory (OOM)!")
        else:
            speedup = latency / latency_p if not oom else float('nan')
            speedup_str = f"{speedup:.2f}x speedup" if not oom else "N/A (Baseline OOM'd!)"
            print(f"    Latency: {latency_p:,.1f} ms | Throughput: {throughput_p:.2f} tokens/sec ({speedup_str})")
            
    # Print comparison table
    print("\n" + "="*60)
    print(" 📈 FINAL BENCHMARK SUMMARY TABLE ")
    print("="*60)
    print(f"{'Prompt Len':<12} | {'Full Latency':<14} | {'Proactive Latency':<18} | {'Speedup':<10}")
    print("-"*60)
    for i, seq_len in enumerate(seq_lengths):
        f_lat = f"{results['full_latency'][i]:,.1f} ms" if not results["full_oom"][i] else "OOM"
        p_lat = f"{results['proactive_latency'][i]:,.1f} ms" if not results["proactive_oom"][i] else "OOM"
        
        if results["full_oom"][i]:
            speedup = "∞ (OOM Avoided!)"
        elif results["proactive_oom"][i]:
            speedup = "N/A"
        else:
            speedup = f"{results['full_latency'][i] / results['proactive_latency'][i]:.2f}x"
            
        print(f"{seq_len:<12} | {f_lat:<14} | {p_lat:<18} | {speedup:<10}")
    print("="*60)

    # 3. Generate Scaling Curve Diagram
    print("\nGenerating scaling curve diagram using Matplotlib...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Latency
    full_l = [l if not results["full_oom"][i] else None for i, l in enumerate(results["full_latency"])]
    proactive_l = [l if not results["proactive_oom"][i] else None for i, l in enumerate(results["proactive_latency"])]
    
    ax1.plot(seq_lengths, full_l, marker='o', color='#d62728', linewidth=2.5, label='Full Attention (Baseline)')
    ax1.plot(seq_lengths, proactive_l, marker='s', color='#1f77b4', linewidth=2.5, label=f'Proactive Cache (budget={budget})')
    
    # Highlight OOMs if any
    for i, seq_len in enumerate(seq_lengths):
        if results["full_oom"][i]:
            ax1.text(seq_len, 500, '❌ OOM', color='#d62728', fontweight='bold', ha='center')
        if results["proactive_oom"][i]:
            ax1.text(seq_len, 500, '❌ OOM', color='#1f77b4', fontweight='bold', ha='center')
            
    ax1.set_title('Generation Latency (50 tokens)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Prompt Sequence Length (tokens)', fontsize=11)
    ax1.set_ylabel('Wall-clock Time (ms)', fontsize=11)
    ax1.set_xticks(seq_lengths)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Plot 2: Throughput
    full_t = [t if not results["full_oom"][i] else 0 for i, t in enumerate(results["full_throughput"])]
    proactive_t = [t if not results["proactive_oom"][i] else 0 for i, t in enumerate(results["proactive_throughput"])]
    
    ax2.plot(seq_lengths, full_t, marker='o', color='#d62728', linewidth=2.5, label='Full Attention')
    ax2.plot(seq_lengths, proactive_t, marker='s', color='#1f77b4', linewidth=2.5, label='Proactive Cache')
    
    ax2.set_title('Generation Throughput (Higher is Better)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Prompt Sequence Length (tokens)', fontsize=11)
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11)
    ax2.set_xticks(seq_lengths)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = "scaling_curve_demo.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f" 🎉 SUCCESS! Scaling curve diagram saved to: {os.path.abspath(save_path)}")
    print("="*60)

if __name__ == "__main__":
    main()

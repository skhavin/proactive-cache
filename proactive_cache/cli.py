import os
import sys
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from proactive_cache.core import ProactiveCache

def find_cached_models():
    """Look up cached models in the local HuggingFace cache folder (~/.cache/huggingface/hub)."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        return []
    
    # Hugging Face caches repositories in folders named "models--author--model-name"
    folders = glob.glob(os.path.join(cache_dir, "models--*"))
    models = []
    for f in folders:
        name = os.path.basename(f)
        parts = name.split("--")
        if len(parts) >= 3:
            author = parts[1]
            model_name = "--".join(parts[2:])
            models.append(f"{author}/{model_name}")
        elif len(parts) == 2:
            models.append(parts[1])
    return sorted(list(set(models)))

def run_easy_mode():
    print("\n" + "="*60)
    print(" 🚀 PROACTIVE CACHE — EASY CALIBRATION & PROTOTYPING ")
    print("="*60)
    print("Scanning your local Hugging Face cache for downloaded models...")
    
    models = find_cached_models()
    
    selected_model_id = None
    if not models:
        print("\n[!] No cached Hugging Face models found in ~/.cache/huggingface/hub.")
        selected_model_id = input("Enter the Hugging Face Model ID you want to use (e.g. Qwen/Qwen2.5-0.5B): ").strip()
    else:
        print("\nFound the following cached models on your local system:")
        for idx, model_id in enumerate(models, 1):
            print(f"  {idx}) {model_id}")
        print(f"  {len(models)+1}) [Enter a custom model ID]")
        
        while True:
            try:
                choice = input(f"\nSelect a model to profile (1-{len(models)+1}): ").strip()
                if not choice:
                    continue
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(models):
                    selected_model_id = models[choice_idx - 1]
                    break
                elif choice_idx == len(models) + 1:
                    selected_model_id = input("Enter custom Hugging Face Model ID: ").strip()
                    break
            except ValueError:
                print("Invalid choice. Please enter a valid number.")
    
    if not selected_model_id:
        print("[!] No model selected. Exiting.")
        return

    print(f"\nLoading model '{selected_model_id}'...")
    print("Note: If the model is not already downloaded, it will be cached locally.")
    
    try:
        # Load in half-precision and map to GPU automatically to fit in VRAM
        tokenizer = AutoTokenizer.from_pretrained(selected_model_id, trust_remote_code=True)
        
        # Load model using appropriate float16 or device map to prevent OOM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            selected_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        print("\n" + "-"*40)
        print(f"Successfully loaded {selected_model_id}!")
        print("-"*40)
        
        # Standardize safe filename for the prototype PKL
        safe_name = selected_model_id.replace("/", "--").replace("\\", "--").lower()
        save_path = f"{safe_name}_prototypes.pkl"
        
        print(f"\nGenerating offline prototype centroids on Wikitext (this may take a moment)...")
        ProactiveCache.profile(
            model=model,
            tokenizer=tokenizer,
            corpus="wikitext",
            num_docs=5,       # Fast profiling for CLI easy mode
            seq_len=512,
            n_clusters=4,
            save_path=save_path
        )
        
        print("\n" + "="*60)
        print(" 🎉 CALIBRATION SUCCESSFUL! PROTOTYPES GENERATED!")
        print("="*60)
        print(f"Prototypes saved to: {os.path.abspath(save_path)}")
        print("\nHere is the exact Python syntax to run this model in O(1) step time:")
        print("-"*60)
        print(f"""from proactive_cache import ProactiveCache
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{selected_model_id}")

# 1. Apply Proactive Cache eviction with a budget of 512 tokens
model = ProactiveCache.apply(
    model, 
    budget=512, 
    prototype_path="{save_path}"
)

# 2. Run inference in O(1) step time!
output = model.generate(input_ids, max_new_tokens=100)""")
        print("-"*60)
        
    except Exception as e:
        print(f"\n[!] Error loading or profiling model: {e}")
        print("Please verify the model ID is correct and fits in your system memory/VRAM.")

def run_info_mode():
    import proactive_cache
    print("\n" + "="*60)
    print(" ℹ️ PROACTIVE CACHE — SYSTEM & LIBRARY INFORMATION ")
    print("="*60)
    print(f"  Library Name:      proactive-cache")
    print(f"  Installed Version: {proactive_cache.__version__}")
    print(f"  License:           GNU Affero General Public License v3 (AGPLv3)")
    print(f"  Package Location:  {os.path.dirname(os.path.abspath(proactive_cache.__file__))}")
    
    print("\n[ Environment Status ]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  CUDA Available:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Active GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Default Device:    {device}")
    
    print("\n[ Local Hugging Face Cache ]")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"  Cache Directory:   {cache_dir}")
    models = find_cached_models()
    print(f"  Cached Models:     {len(models)}")
    if models:
        for idx, model_id in enumerate(models, 1):
            print(f"    - {model_id}")
            
    print("\n[ Authors & References ]")
    print(f"  Author:            {proactive_cache.__author__}")
    print(f"  GitHub Code:       https://github.com/skhavin/proactive-cache")
    print(f"  Methodology Repo:  https://github.com/skhavin/supertransformers")
    
    print("\n[ CLI Commands ]")
    print(f"  proactive-cache --easy    Run interactive local calibration and prototyping.")
    print(f"  proactive-cache --info    Display this environment information screen.")
    print("="*60 + "\n")

def main():
    if "--easy" in sys.argv:
        run_easy_mode()
    elif "--info" in sys.argv:
        run_info_mode()
    else:
        print("\nProactive Cache Command-Line Tool")
        print("Usage:")
        print("  proactive-cache --easy      Scan local HuggingFace cache and run automated calibration.")
        print("  proactive-cache --info      Display detailed system and library information.")

if __name__ == "__main__":
    main()

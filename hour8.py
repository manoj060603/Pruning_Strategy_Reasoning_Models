import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import time
import numpy as np

# Suppress warnings for clean logs
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION (OPTIMIZED SETTINGS)
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 1200       # High limit for complex reasoning
TEMPERATURE = 0.6           # Balanced creativity/logic
PRUNING_TRIGGER_COUNT = 4   # Check for pruning every 4 steps
PRUNING_RATIO = 0.5         # Only prune if score < 50% of the Best Chunk's score
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

# ==========================================
# 1. SETUP
# ==========================================
print("--- Hour 8: Debugging & Optimization (Stable Build) ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Force 'eager' for attention access
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

# Pre-compute probe
probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print("System Ready.")

# ==========================================
# 2. ROBUST STEP DETECTION
# ==========================================
SPLIT_WORDS = ["Wait", "Alternatively", "Therefore", "So", "Thus", "However", "But", "Let", "First", "Next", "Finally", "Now", "We", "Recall", "Notice", "Check"]
split_token_ids = set()
for word in SPLIT_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids: split_token_ids.add(ids[0])
    ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
    if ids_space: split_token_ids.add(ids_space[0])
nl = tokenizer.encode("\n", add_special_tokens=False)
if nl: split_token_ids.add(nl[-1])

def is_step_boundary(token_id):
    return token_id in split_token_ids

# ==========================================
# 3. FORKED ATTENTION PROBE
# ==========================================
def get_attention_vector(current_input_ids):
    """
    Stateless probe: Forks input -> Appends Probe -> Gets Attention -> Discards Fork
    """
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    
    with torch.no_grad():
        outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
    
    # Last Layer, Average Heads, Last Token (Probe End) -> All History
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1)
    final_token_attn = avg_attn[0, -1, :]
    
    return final_token_attn[:current_input_ids.shape[1]]

# ==========================================
# 4. PRUNE & REFRESH (OPTIMIZED)
# ==========================================
def prune_and_refresh(chunks_metadata, attn_vector, full_input_ids, tokenizer):
    # 1. Score chunks
    scored_chunks = []
    for i, meta in enumerate(chunks_metadata):
        start, end = meta["start"], meta["end"]
        if end > len(attn_vector): end = len(attn_vector)
        
        # Calculate mean score for this chunk
        score = attn_vector[start:end].mean().item()
        
        scored_chunks.append({
            "meta": meta,
            "score": score,
            "keep": True,
            "is_active": (i == len(chunks_metadata) - 1)
        })

    # 2. Candidate Selection
    candidates = [c for c in scored_chunks if not c["is_active"]]
    if not candidates: 
        return full_input_ids, False, 0
    
    # 3. Safety Threshold Logic
    best_chunk_score = max(c["score"] for c in candidates)
    victim = min(candidates, key=lambda x: x["score"])
    
    # OPTIMIZATION: Only prune if the chunk is significantly worse than the best one.
    # This prevents pruning when all chunks are actually important.
    threshold = best_chunk_score * PRUNING_RATIO
    
    if victim["score"] < threshold:
        victim["keep"] = False
        
        print(f"\n   >>> [PRUNE] \"{victim['meta']['text'].strip()[:30]}...\" (Score: {victim['score']:.4f} < Threshold: {threshold:.4f})")
        
        # 4. Reconstruct Text
        # Recover the Prompt (Pre-Chunk Text)
        first_chunk_start = chunks_metadata[0]["start"]
        prompt_ids = full_input_ids[0, :first_chunk_start]
        prompt_text = tokenizer.decode(prompt_ids) # This might include special tokens depending on decoding
        
        # Concatenate Kept Chunks
        kept_body = "".join([c["meta"]["text"] for c in scored_chunks if c["keep"]])
        full_new_text = prompt_text + kept_body
        
        # 5. Context Refresh (Re-Tokenize)
        # We assume the chat template special tokens are handled by the tokenizer's default behavior
        # or preserved in the decoded string if skip_special_tokens=False
        new_inputs = tokenizer(full_new_text, return_tensors="pt").to(full_input_ids.device)
        
        saved = full_input_ids.shape[1] - new_inputs.input_ids.shape[1]
        return new_inputs.input_ids, True, saved
    
    return full_input_ids, False, 0

# ==========================================
# 5. MAIN EXECUTION LOOP (STABLE)
# ==========================================
def run_stable_experiment(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*60}")
    
    # Init
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    
    input_ids = inputs.input_ids
    past_key_values = None
    
    # State Tracking
    chunk_start_idx = input_ids.shape[1]
    chunks_metadata = []
    total_pruned = 0
    
    # Performance Tracking
    t0 = time.time()
    generated_tokens_count = 0
    probe_time_total = 0
    
    print("Generating: ", end="", flush=True)
    
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            # A. Generate
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens_count += 1
            
            # Print
            word = tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
            # Update
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            
            # B. Boundary Check
            if is_step_boundary(next_token_id):
                chunk_end_idx = input_ids.shape[1]
                chunk_text = tokenizer.decode(input_ids[0, chunk_start_idx:chunk_end_idx])
                
                chunks_metadata.append({
                    "start": chunk_start_idx, 
                    "end": chunk_end_idx, 
                    "text": chunk_text
                })
                
                # C. Pruning Trigger
                if len(chunks_metadata) >= PRUNING_TRIGGER_COUNT:
                    p_start = time.time()
                    
                    # 1. Probe
                    attn_vec = get_attention_vector(input_ids)
                    
                    # 2. Prune
                    new_ids, pruned, saved = prune_and_refresh(chunks_metadata, attn_vec, input_ids, tokenizer)
                    
                    p_end = time.time()
                    probe_time_total += (p_end - p_start)
                    
                    if pruned:
                        print(f" [REFRESH: -{saved} T] ", end="", flush=True)
                        total_pruned += saved
                        
                        # Apply State Update
                        input_ids = new_ids
                        past_key_values = None # CRITICAL: Flush Cache
                        
                        # Reset tracking
                        chunk_start_idx = input_ids.shape[1]
                        chunks_metadata = []
                        continue 
                
                chunk_start_idx = chunk_end_idx
            
            if next_token_id == tokenizer.eos_token_id:
                print("\n\n[EOS Reached]")
                break
                
    total_time = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"1. Total Tokens Generated: {generated_tokens_count}")
    print(f"2. Total Tokens Pruned:    {total_pruned} ({(total_pruned/(total_pruned+generated_tokens_count))*100:.1f}% reduction)")
    print(f"3. Final Context Size:     {input_ids.shape[1]}")
    print(f"4. Total Time:             {total_time:.2f}s")
    print(f"5. Latency (incl. probes): {generated_tokens_count / total_time:.2f} tokens/sec")
    print(f"6. Probe Overhead:         {probe_time_total:.2f}s total")
    print(f"{'='*60}")

# ==========================================
# 6. EXECUTE HARD TEST
# ==========================================
# This problem usually causes models to loop between combinations vs permutations
hard_q = "A committee of 5 people is to be chosen from a group of 8 men and 6 women. What is the probability that the committee contains at least 3 men? Verify your answer by calculating the complement."
run_stable_experiment(hard_q)
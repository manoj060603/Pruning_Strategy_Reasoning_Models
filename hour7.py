import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import time

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
# The "Probe" from the paper that forces the model to summarize importance
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"
PRUNING_TRIGGER_COUNT = 4  # Prune after every 4 chunks accumulate
PRUNING_AGGRESSIVENESS = 0.6 # Prune if score is < 60% of the best chunk

# ==========================================
# 1. SETUP & LOADING
# ==========================================
print("--- Hour 7: Integrated Think Clearly Pipeline ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Critical: attn_implementation="eager" is required to access attention weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

# Pre-compute probe tensors
probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print("Pipeline Ready.")

# ==========================================
# 2. STEP DETECTION LOGIC
# ==========================================
SPLIT_WORDS = ["Wait", "Alternatively", "Therefore", "So", "Thus", "However", "But", "Let", "First", "Next", "Finally", "Now", "We", "Recall"]
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
# 3. PROBING LOGIC
# ==========================================
def get_attention_vector(current_input_ids):
    """
    Forks the input, runs the probe, and returns attention scores for the history.
    """
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    
    with torch.no_grad():
        # use_cache=False forces full re-computation for accurate attention
        outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
    
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1) # Average heads
    final_token_attn = avg_attn[0, -1, :]   # Vector from Probe-End to History
    
    # Return slice corresponding to original history
    return final_token_attn[:current_input_ids.shape[1]]

# ==========================================
# 4. PRUNING LOGIC (CONTEXT REFRESH)
# ==========================================
def prune_and_refresh(chunks_metadata, attn_vector, full_input_ids, tokenizer):
    """
    Identifies the weak chunk, reconstructs text, and returns new inputs.
    """
    # 1. Score Chunks
    scored_chunks = []
    for i, meta in enumerate(chunks_metadata):
        start, end = meta["start"], meta["end"]
        if end > len(attn_vector): end = len(attn_vector)
        score = attn_vector[start:end].mean().item()
        
        scored_chunks.append({
            "meta": meta,
            "score": score,
            "keep": True,
            "is_active": (i == len(chunks_metadata) - 1) # Never prune the last active chunk
        })

    # 2. Identify Candidates
    candidates = [c for c in scored_chunks if not c["is_active"]]
    if not candidates: 
        return full_input_ids, False, 0
    
    # 3. Pruning Strategy
    best_score = max(c["score"] for c in candidates)
    victim = min(candidates, key=lambda x: x["score"])
    
    # Prune if the victim is significantly worse than the best (TRAAC-inspired relative threshold)
    if victim["score"] < (best_score * PRUNING_AGGRESSIVENESS):
        victim["keep"] = False
        
        # Log the action
        print(f"\n\n   >>> [PRUNING] Removing: \"{victim['meta']['text'].strip()[:40]}...\"")
        print(f"   >>> Stats: Score {victim['score']:.5f} (Best was {best_score:.5f})")
        
        # 4. Reconstruct Text
        first_chunk_start = chunks_metadata[0]["start"]
        prompt_ids = full_input_ids[0, :first_chunk_start]
        prompt_text = tokenizer.decode(prompt_ids)
        
        new_body_text = ""
        for chunk in scored_chunks:
            if chunk["keep"]:
                new_body_text += chunk["meta"]["text"]
                
        full_new_text = prompt_text + new_body_text
        
        # 5. Re-Tokenize (Context Refresh)
        new_inputs = tokenizer(full_new_text, return_tensors="pt").to(full_input_ids.device)
        
        tokens_saved = full_input_ids.shape[1] - new_inputs.input_ids.shape[1]
        return new_inputs.input_ids, True, tokens_saved
    
    return full_input_ids, False, 0

# ==========================================
# 5. THE INTEGRATED GENERATION LOOP
# ==========================================
def generate_think_clearly(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*60}")
    
    # Prepare Initial Inputs
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    
    input_ids = inputs.input_ids
    past_key_values = None # KV Cache starts empty
    
    # Metadata Tracking
    prompt_len = input_ids.shape[1]
    chunk_start_idx = prompt_len
    chunks_metadata = []
    
    total_pruned_tokens = 0
    start_time = time.time()
    
    print("Generating: ", end="", flush=True)
    
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            # -------------------------------------------------
            # A. GENERATION STEP
            # -------------------------------------------------
            # If cache is empty (start or after prune), use full inputs.
            # If cache exists, use only the last token.
            if past_key_values is None:
                model_inputs = input_ids
            else:
                model_inputs = input_ids[:, -1:]
                
            outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decoding for demo stability
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Print token
            word = tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
            # Update tensors
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            
            # -------------------------------------------------
            # B. BOUNDARY CHECK
            # -------------------------------------------------
            if is_step_boundary(next_token_id):
                chunk_end_idx = input_ids.shape[1]
                
                # Capture chunk text
                # We perform this decode only on boundary to save time
                chunk_ids = input_ids[0, chunk_start_idx:chunk_end_idx]
                chunk_text = tokenizer.decode(chunk_ids)
                
                chunks_metadata.append({
                    "start": chunk_start_idx, 
                    "end": chunk_end_idx, 
                    "text": chunk_text
                })
                
                # -------------------------------------------------
                # C. PROBE & PRUNE LOGIC
                # -------------------------------------------------
                # Only check if we have enough context chunks
                if len(chunks_metadata) >= PRUNING_TRIGGER_COUNT:
                    
                    # 1. Run Probe (Slow, but necessary)
                    attn_vec = get_attention_vector(input_ids)
                    
                    # 2. Run Pruning Logic
                    new_ids, pruned, saved = prune_and_refresh(chunks_metadata, attn_vec, input_ids, tokenizer)
                    
                    if pruned:
                        print(f" [REFRESH: -{saved} Tokens] ", end="", flush=True)
                        total_pruned_tokens += saved
                        
                        # CRITICAL UPDATE
                        input_ids = new_ids
                        past_key_values = None # Force Cache Flush
                        
                        # Reset Metadata Tracking
                        # We restart tracking from the new end position
                        prompt_len = input_ids.shape[1]
                        chunk_start_idx = prompt_len
                        chunks_metadata = []
                        continue # Skip the normal index update
                
                # Update start index for next chunk
                chunk_start_idx = chunk_end_idx
            
            # -------------------------------------------------
            # D. STOP CONDITION
            # -------------------------------------------------
            if next_token_id == tokenizer.eos_token_id:
                print("\n\n[EOS Reached]")
                break
                
    end_time = time.time()
    
    # Final Report
    print(f"\n{'='*60}")
    print(f"Total Tokens Generated (Raw): {i+1}")
    print(f"Total Tokens Pruned:        {total_pruned_tokens}")
    print(f"Final Context Length:       {input_ids.shape[1]}")
    print(f"Effective Reduction:        {total_pruned_tokens / (i+1 + total_pruned_tokens) * 100:.2f}%")
    print(f"Time Elapsed:               {end_time - start_time:.2f}s")
    print(f"{'='*60}")

# ==========================================
# 6. EXECUTE TEST
# ==========================================
# A prompt designed to trigger "Self-Correction" (common target for pruning)
test_prompt = "Calculate the sum of the coefficients of (3x - 2)^4. Expand it fully first to verify."
generate_think_clearly(test_prompt)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 512
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

# ==========================================
# 1. SETUP
# ==========================================
print("--- Hour 6: Context Refresh Pruning Strategy ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Force 'eager' attention so we can extract weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

# Pre-compute probe tensor
probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print("Model loaded. Pruning Logic Ready.")

# ==========================================
# 2. HELPER: STEP DETECTION (From Hour 2)
# ==========================================
SPLIT_WORDS = ["Wait", "Alternatively", "Therefore", "So", "Thus", "However", "But", "Let", "First", "Next", "Finally", "Now", "We"]
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
# 3. HELPER: ATTENTION PROBE (From Hour 4)
# ==========================================
def get_attention_vector(current_input_ids):
    # Fork
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    
    # Forward (No Cache)
    with torch.no_grad():
        outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
    
    # Extract Last Layer Attention
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1) # Avg across heads
    
    # Vector from Probe-End to History
    final_token_attn = avg_attn[0, -1, :]
    
    # Return valid history slice
    return final_token_attn[:current_input_ids.shape[1]]

# ==========================================
# 4. THE CORE LOGIC: PRUNE AND REFRESH
# ==========================================
def prune_and_refresh(chunks_metadata, attn_vector, full_input_ids, tokenizer):
    """
    1. Calculates scores for each chunk.
    2. Identifies the lowest scoring chunk (excluding the active one).
    3. Reconstructs text WITHOUT that chunk.
    4. Re-tokenizes to create fresh input_ids.
    """
    
    # A. Score the Chunks
    scored_chunks = []
    for i, meta in enumerate(chunks_metadata):
        start, end = meta["start"], meta["end"]
        # Safety clamp
        if end > len(attn_vector): end = len(attn_vector)
        
        # Mean attention for this chunk
        score = attn_vector[start:end].mean().item()
        
        scored_chunks.append({
            "meta": meta,
            "score": score,
            "keep": True, # Default keep
            "is_active": (i == len(chunks_metadata) - 1) # Don't prune the one we just finished writing
        })

    # B. Identify Victim
    # Filter out active chunk so we don't break flow
    candidates = [c for c in scored_chunks if not c["is_active"]]
    
    if not candidates:
        return full_input_ids, False, 0 # Nothing to prune yet
    
    # Find lowest score
    victim = min(candidates, key=lambda x: x["score"])
    
    # C. Execute Pruning (Mark as False)
    # We prune if we have candidates. In real exp, apply threshold (e.g., score < 0.001)
    victim["keep"] = False
    
    print(f"\n   >>> [PRUNING ACTION] Removing Chunk: \"{victim['meta']['text'].strip()[:30]}...\" (Score: {victim['score']:.5f})")
    
    # D. Reconstruct Text
    # 1. Get Prompt (Everything before first chunk)
    first_chunk_start = chunks_metadata[0]["start"]
    prompt_ids = full_input_ids[0, :first_chunk_start]
    prompt_text = tokenizer.decode(prompt_ids)
    
    # 2. Append Kept Chunks
    new_body_text = ""
    for chunk in scored_chunks:
        if chunk["keep"]:
            new_body_text += chunk["meta"]["text"]
            
    full_new_text = prompt_text + new_body_text
    
    # E. Re-Tokenize (Context Refresh)
    new_inputs = tokenizer(full_new_text, return_tensors="pt").to(full_input_ids.device)
    new_input_ids = new_inputs.input_ids
    
    # Calculate savings
    tokens_removed = full_input_ids.shape[1] - new_input_ids.shape[1]
    
    return new_input_ids, True, tokens_removed

# ==========================================
# 5. EXECUTION DEMO
# ==========================================
def run_pruning_demo(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*50}")
    print("Generating with Context Refresh Pruning...\n")
    
    # Init inputs
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Init State
    past_key_values = None
    prompt_len = input_ids.shape[1]
    chunk_start_idx = prompt_len
    chunks_metadata = []
    
    total_pruned = 0
    
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            # 1. Standard Generation
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            
            # Print
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            
            # Update history
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            
            # 2. Check Boundary
            if is_step_boundary(next_token_id):
                chunk_end_idx = input_ids.shape[1]
                
                # Record Chunk
                chunk_ids = input_ids[0, chunk_start_idx:chunk_end_idx]
                chunk_text = tokenizer.decode(chunk_ids)
                chunks_metadata.append({"start": chunk_start_idx, "end": chunk_end_idx, "text": chunk_text})
                
                # 3. TRIGGER PRUNING (Every 3 chunks for demo)
                if len(chunks_metadata) >= 3:
                    # Get Attention
                    attn_vec = get_attention_vector(input_ids)
                    
                    # Call the Deliverable Function
                    new_ids, pruned, saved = prune_and_refresh(chunks_metadata, attn_vec, input_ids, tokenizer)
                    
                    if pruned:
                        print(f" [REFRESH: -{saved} Tok] ", end="", flush=True)
                        total_pruned += saved
                        
                        # UPDATE STATE (The "Fix")
                        input_ids = new_ids
                        past_key_values = None # Flush Cache!
                        
                        # Reset Metadata (Simplification for demo)
                        # We clear metadata because indices shifted. Tracking restarts from here.
                        chunk_start_idx = input_ids.shape[1]
                        chunks_metadata = [] 
                        continue # Skip updating chunk_start_idx at bottom
                
                chunk_start_idx = chunk_end_idx

            if next_token_id == tokenizer.eos_token_id:
                print("\n[EOS]")
                break
                
    print(f"\n\nTotal Tokens Pruned: {total_pruned}")

# ==========================================
# 6. RUN
# ==========================================
test_q = "Explain the process of photosynthesis in detail, including light-dependent reactions."
run_pruning_demo(test_q)
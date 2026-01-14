import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

# ==========================================
# 1. SETUP
# ==========================================
print("--- Hour 5: Scoring Aggregation ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Force 'eager' to allow attention extraction
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

# Prepare Probe Tensor
probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print("Model Loaded. Ready to Score.")

# ==========================================
# 2. STEP DETECTION
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
# 3. PROBE FUNCTION (Stateless)
# ==========================================
def get_attention_vector(current_input_ids):
    """
    Returns the attention profile of the history based on the probe.
    """
    # 1. Fork input with probe
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    
    # 2. Forward pass (No Cache)
    with torch.no_grad():
        outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
    
    # 3. Extract Attention
    # Last Layer, Average Heads
    last_layer_attn = outputs.attentions[-1] # [Batch, Heads, Seq, Seq]
    avg_attn = last_layer_attn.mean(dim=1)   # [Batch, Seq, Seq]
    
    # Vector: Attention FROM the last probe token TO the history
    # Batch 0, Row -1
    final_token_attn = avg_attn[0, -1, :]
    
    # Slice only the original history (ignore attention to the probe itself)
    history_len = current_input_ids.shape[1]
    return final_token_attn[:history_len]

# ==========================================
# 4. SCORING AGGREGATION LOOP
# ==========================================
def run_scoring_experiment(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*50}")
    print("Generating with Real-Time Importance Scoring...\n")
    
    # Inputs
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Tracking State
    past_key_values = None
    
    # We need to track where chunks begin and end
    # Chunks start AFTER the prompt
    prompt_len = input_ids.shape[1]
    chunk_start_idx = prompt_len
    
    # List to store (Start_Idx, End_Idx, Text_Content)
    chunks_metadata = [] 
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # A. Generate Token
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decoding for consistency in demo
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Update History
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            token_str = tokenizer.decode([next_token_id])
            print(token_str, end="", flush=True)
            
            # B. CHECK BOUNDARY
            if is_step_boundary(next_token_id):
                chunk_end_idx = input_ids.shape[1]
                
                # decoding the specific chunk text for visualization
                # We slice input_ids[0, start:end]
                chunk_ids = input_ids[0, chunk_start_idx:chunk_end_idx]
                chunk_text = tokenizer.decode(chunk_ids)
                
                # Save Metadata
                chunks_metadata.append({
                    "start": chunk_start_idx,
                    "end": chunk_end_idx,
                    "text": chunk_text
                })
                
                # --- SCORING LOGIC ---
                # 1. Get Global Attention Vector
                attn_vector = get_attention_vector(input_ids)
                
                # 2. Score Each Chunk
                scored_chunks = []
                for meta in chunks_metadata:
                    # Extract values from vector
                    c_start, c_end = meta["start"], meta["end"]
                    
                    # Safety check
                    if c_end > len(attn_vector): c_end = len(attn_vector)
                    
                    c_attn = attn_vector[c_start:c_end]
                    score = c_attn.mean().item()
                    scored_chunks.append((meta["text"], score))
                
                # 3. VISUAL CHECK
                # Sort by score (ascending)
                scored_chunks.sort(key=lambda x: x[1])
                
                lowest_chunk_text, lowest_score = scored_chunks[0]
                highest_chunk_text, highest_score = scored_chunks[-1]
                
                # Print Debug Info
                print("\n" + "-"*40)
                print(f" [STEP {len(chunks_metadata)} COMPLETE]")
                print(f"  > High Score ({highest_score*1000:.2f}): \"{highest_chunk_text.strip()[:40]}...\"")
                print(f"  > Low Score  ({lowest_score*1000:.2f}):  \"{lowest_chunk_text.strip()[:40]}...\"")
                
                # Check if lowest looks like fluff
                print(f"  >> CANDIDATE FOR PRUNING: \"{lowest_chunk_text.strip()}\"")
                print("-" * 40 + "\n")
                
                # Reset for next chunk
                chunk_start_idx = chunk_end_idx
            
            if next_token_id == tokenizer.eos_token_id:
                print("\n[EOS]")
                break

    return chunks_metadata

# ==========================================
# 5. EXECUTE
# ==========================================
# Prompt engineered to likely cause backtracking/verification steps
test_q = "Calculate the sum of integers x such that |x - 2| < 4. Double check your work."
run_scoring_experiment(test_q)
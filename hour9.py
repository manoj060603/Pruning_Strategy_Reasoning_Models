import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
import time
import json
import re
import numpy as np
import gc

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION: EXPERIMENT A (WINDOWED)
# ==========================================
EXPERIMENT_NAME = "Experiment_A_Windowed"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_ID = "HuggingFaceH4/aime_2024"
NUM_SAMPLES = 10           
MAX_NEW_TOKENS = 8192       
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

PRUNING_TRIGGER_COUNT = 4   
PRUNING_RATIO = 0.25
PROBE_WINDOW_SIZE = 2500  # <--- NEW SAFETY LIMIT (Keep this under 3000 for A100/3090)

# ==========================================
# 1. SETUP
# ==========================================
print(f"--- Hour 9: {EXPERIMENT_NAME} ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print(f"Model Loaded. Window Safety Limit: {PROBE_WINDOW_SIZE} tokens")

# ==========================================
# 2. HELPER FUNCTIONS
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

# --- WINDOWED PROBE FUNCTION ---
def get_attention_vector_windowed(full_input_ids):
    """
    Probes only the last PROBE_WINDOW_SIZE tokens to avoid OOM.
    Returns: (attention_vector, window_start_index)
    """
    seq_len = full_input_ids.shape[1]
    
    # Determine Window
    if seq_len > PROBE_WINDOW_SIZE:
        window_start = seq_len - PROBE_WINDOW_SIZE
        # Slice the input: Keep first part (Prompt) + Window? 
        # No, just take the window. The attention logic holds locally.
        input_slice = full_input_ids[:, window_start:]
    else:
        window_start = 0
        input_slice = full_input_ids
    
    try:
        torch.cuda.empty_cache()
        
        # Attach Probe
        probe_part = probe_tensor_template.to(input_slice.device)
        forked_input_ids = torch.cat([input_slice, probe_part], dim=1)
        
        with torch.no_grad():
            outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
        
        last_layer_attn = outputs.attentions[-1]
        avg_attn = last_layer_attn.mean(dim=1)
        final_token_attn = avg_attn[0, -1, :]
        
        # Extract Result (CPU)
        # Valid length is length of input_slice
        result = final_token_attn[:input_slice.shape[1]].clone().cpu()
        
        # Clean up
        del outputs, last_layer_attn, avg_attn, final_token_attn, forked_input_ids
        torch.cuda.empty_cache()
        
        return result, window_start
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, -1

def extract_answer(text):
    match = re.findall(r"\\(?:boxed|framebox|textbf)\{?\(?([0-9\.]+)\)?\}?", text)
    if match: return match[-1]
    match_text = re.findall(r"(?:The answer is|final answer is)\s*(\d+)", text, re.IGNORECASE)
    if match_text: return match_text[-1]
    return None

def normalize_answer(ans):
    if not ans: return ""
    try:
        clean = re.sub(r"[,\s\$]", "", ans)
        return str(float(clean))
    except:
        return ans.strip()

# ==========================================
# 3. PRUNING LOGIC (OFFSET AWARE)
# ==========================================
def prune_and_refresh(chunks_metadata, attn_vector, window_start, full_input_ids, tokenizer):
    scored_chunks = []
    
    for i, meta in enumerate(chunks_metadata):
        # We need to check if this chunk falls inside our window
        c_start, c_end = meta["start"], meta["end"]
        
        # Relative coordinates inside the window
        rel_start = c_start - window_start
        rel_end = c_end - window_start
        
        # Only score if fully inside window
        if rel_start >= 0 and rel_end <= len(attn_vector):
            score = attn_vector[rel_start:rel_end].mean().item()
            scored_chunks.append({
                "meta": meta, "score": score, "keep": True, 
                "is_active": (i == len(chunks_metadata) - 1)
            })
            
    # Need candidates
    candidates = [c for c in scored_chunks if not c["is_active"]]
    if not candidates: return full_input_ids, False, 0
    
    best_chunk_score = max(c["score"] for c in candidates)
    victim = min(candidates, key=lambda x: x["score"])
    
    threshold = best_chunk_score * PRUNING_RATIO
    
    if victim["score"] < threshold:
        victim["keep"] = False
        
        # Reconstruct Text
        # 1. Prompt (Everything up to first chunk in METADATA, not window)
        first_meta_start = chunks_metadata[0]["start"]
        prompt_ids = full_input_ids[0, :first_meta_start]
        prompt_text = tokenizer.decode(prompt_ids)
        
        # 2. Rebuild body from chunks_metadata
        # (We iterate over ALL chunks. If it was in window and marked prune, skip it. If outside window, keep it.)
        new_body_text = ""
        victim_start = victim["meta"]["start"]
        
        for meta in chunks_metadata:
            # If this is the victim, skip
            if meta["start"] == victim_start:
                continue
            new_body_text += meta["text"]
            
        full_new_text = prompt_text + new_body_text
        
        new_inputs = tokenizer(full_new_text, return_tensors="pt").to(full_input_ids.device)
        saved = full_input_ids.shape[1] - new_inputs.input_ids.shape[1]
        
        return new_inputs.input_ids, True, saved
    
    return full_input_ids, False, 0

# ==========================================
# 4. GENERATION LOOP (ROBUST)
# ==========================================
def run_pipeline(prompt_text, q_id):
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    past_key_values = None
    
    chunk_start_idx = input_ids.shape[1]
    chunks_metadata = []
    total_pruned = 0
    t0 = time.time()
    truncated = True 
    oom_events = 0
    
    print(f"Problem {q_id}: Generating", end="", flush=True)
    
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            try:
                if past_key_values is None:
                    outputs = model(input_ids, use_cache=True)
                else:
                    outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                
                past_key_values = outputs.past_key_values
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
                
                if is_step_boundary(next_token_id):
                    chunk_end_idx = input_ids.shape[1]
                    chunk_text = tokenizer.decode(input_ids[0, chunk_start_idx:chunk_end_idx])
                    chunks_metadata.append({"start": chunk_start_idx, "end": chunk_end_idx, "text": chunk_text})
                    
                    if len(chunks_metadata) >= PRUNING_TRIGGER_COUNT:
                        # Call Windowed Probe
                        attn_vec, win_start = get_attention_vector_windowed(input_ids)
                        
                        if attn_vec is not None:
                            new_ids, pruned, saved = prune_and_refresh(chunks_metadata, attn_vec, win_start, input_ids, tokenizer)
                            
                            if pruned:
                                total_pruned += saved
                                input_ids = new_ids
                                past_key_values = None
                                chunk_start_idx = input_ids.shape[1]
                                chunks_metadata = []
                                print(".", end="", flush=True)
                                continue
                        else:
                            oom_events += 1
                            print("x", end="", flush=True)
                    
                    chunk_start_idx = chunk_end_idx
                
                if next_token_id == tokenizer.eos_token_id:
                    truncated = False
                    break
            except torch.cuda.OutOfMemoryError:
                # Emergency catch
                print(" [FATAL OOM] ", end="")
                torch.cuda.empty_cache()
                break
                
    time_taken = time.time() - t0
    final_text = tokenizer.decode(input_ids[0])
    
    if truncated: print(" [TRUNCATED]", end="")
    
    return {
        "final_text": final_text,
        "tokens_generated": i + 1,
        "tokens_pruned": total_pruned,
        "time": time_taken
    }

# ==========================================
# 5. MAIN LOOP
# ==========================================
print("\n--- Loading Dataset (AIME 2024) ---")
dataset = load_dataset(DATASET_ID, split="train").select(range(NUM_SAMPLES))

results = []

for idx, example in enumerate(dataset):
    print(f"\n[{idx+1}/{NUM_SAMPLES}] ", end="")
    torch.cuda.empty_cache()
    gc.collect()
    
    metrics = run_pipeline(example['problem'], idx+1)
    
    model_ans = extract_answer(metrics['final_text'])
    gt_ans = extract_answer(example['solution'])
    is_correct = normalize_answer(model_ans) == normalize_answer(gt_ans)
    
    entry = {
        "id": idx,
        "correct": is_correct,
        "model_answer": model_ans,
        "gt_answer": gt_ans,
        "tokens_total": metrics["tokens_generated"],
        "tokens_pruned": metrics["tokens_pruned"],
        "reduction_pct": (metrics["tokens_pruned"] / (metrics["tokens_generated"] + metrics["tokens_pruned"])) * 100 if metrics["tokens_pruned"] > 0 else 0
    }
    results.append(entry)
    
    print(f"\n  > Pruned: {entry['tokens_pruned']} tokens ({entry['reduction_pct']:.1f}%)")
    print(f"  > Ans: {model_ans} | GT: {gt_ans} | Correct: {is_correct}")

# ==========================================
# 6. RESULTS
# ==========================================
accuracy = sum(r['correct'] for r in results) / len(results) * 100
avg_reduction = np.mean([r['reduction_pct'] for r in results])

print(f"\n\n{'='*40}")
print(f"EXPERIMENT A (WINDOWED) RESULTS")
print(f"{'='*40}")
print(f"Accuracy:        {accuracy:.1f}%")
print(f"Avg Reduction:   {avg_reduction:.1f}%")
print(f"{'='*40}")

with open("results_exp_a_windowed.json", "w") as f:
    json.dump(results, f, indent=4)
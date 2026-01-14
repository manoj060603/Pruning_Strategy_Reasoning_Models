import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 1500  # Increased for rigorous reasoning
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

# ==========================================
# 1. SETUP
# ==========================================
print("--- STRESS TEST: Rigorous Math Problem ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager"
)

probe_tensor_template = torch.tensor([tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)], dtype=torch.long)
print("Ready.")

# ==========================================
# 2. LOGIC (Reusing Hour 6 Logic)
# ==========================================
SPLIT_WORDS = ["Wait", "Alternatively", "Therefore", "So", "Thus", "However", "But", "Let", "First", "Next", "Finally", "Now", "We", "Recall", "Notice"]
split_token_ids = set()
for word in SPLIT_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids: split_token_ids.add(ids[0])
    ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
    if ids_space: split_token_ids.add(ids_space[0])
nl = tokenizer.encode("\n", add_special_tokens=False)
if nl: split_token_ids.add(nl[-1])

def is_step_boundary(t_id): return t_id in split_token_ids

def get_attention_vector(current_input_ids):
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    with torch.no_grad():
        outputs = model(forked_input_ids, use_cache=False, output_attentions=True)
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1)
    final_token_attn = avg_attn[0, -1, :]
    return final_token_attn[:current_input_ids.shape[1]]

def prune_and_refresh(chunks_metadata, attn_vector, full_input_ids, tokenizer):
    # Score chunks
    scored_chunks = []
    for i, meta in enumerate(chunks_metadata):
        start, end = meta["start"], meta["end"]
        if end > len(attn_vector): end = len(attn_vector)
        score = attn_vector[start:end].mean().item()
        scored_chunks.append({
            "meta": meta,
            "score": score,
            "keep": True,
            "is_active": (i == len(chunks_metadata) - 1)
        })

    # Identify victim (Lowest score that isn't active)
    candidates = [c for c in scored_chunks if not c["is_active"]]
    if not candidates: return full_input_ids, False, 0
    
    victim = min(candidates, key=lambda x: x["score"])
    
    # Pruning Threshold Logic
    # For this stress test, we prune if score is in the bottom 50% relative to the best chunk
    # This prevents pruning "good" chunks just because they are the lowest of a good bunch
    best_score = max([c["score"] for c in candidates])
    
    if victim["score"] < (best_score * 0.6): # Aggressive pruning for demo
        victim["keep"] = False
        print(f"\n   >>> [PRUNING] \"{victim['meta']['text'].strip()[:40]}...\" (Score: {victim['score']:.5f} vs Best: {best_score:.5f})")
        
        # Reconstruct
        first_chunk_start = chunks_metadata[0]["start"]
        prompt_ids = full_input_ids[0, :first_chunk_start]
        prompt_text = tokenizer.decode(prompt_ids)
        
        new_body_text = ""
        for chunk in scored_chunks:
            if chunk["keep"]:
                new_body_text += chunk["meta"]["text"]
                
        full_new_text = prompt_text + new_body_text
        new_inputs = tokenizer(full_new_text, return_tensors="pt").to(full_input_ids.device)
        
        return new_inputs.input_ids, True, full_input_ids.shape[1] - new_inputs.input_ids.shape[1]
    
    return full_input_ids, False, 0

# ==========================================
# 3. EXECUTE STRESS TEST
# ==========================================
def run_stress_test(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*50}")
    
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    past_key_values = None
    prompt_len = input_ids.shape[1]
    chunk_start_idx = prompt_len
    chunks_metadata = [] 
    
    total_pruned = 0
    
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            # Generate
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            
            # Print token
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            
            # Update tensors
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            
            # Boundary Check
            if is_step_boundary(next_token_id):
                chunk_end_idx = input_ids.shape[1]
                chunk_text = tokenizer.decode(input_ids[0, chunk_start_idx:chunk_end_idx])
                chunks_metadata.append({"start": chunk_start_idx, "end": chunk_end_idx, "text": chunk_text})
                
                # Prune every 4 chunks to allow context to build
                if len(chunks_metadata) >= 4:
                    attn_vec = get_attention_vector(input_ids)
                    new_ids, pruned, saved = prune_and_refresh(chunks_metadata, attn_vec, input_ids, tokenizer)
                    
                    if pruned:
                        print(f" [REFRESH: -{saved} Tok] ", end="", flush=True)
                        total_pruned += saved
                        input_ids = new_ids
                        past_key_values = None
                        chunk_start_idx = input_ids.shape[1]
                        chunks_metadata = [] 
                        continue
                
                chunk_start_idx = chunk_end_idx

            if next_token_id == tokenizer.eos_token_id:
                print("\n\n[EOS]")
                break
                
    print(f"\n\n{'='*50}")
    print(f"Final Tokens Pruned: {total_pruned}")

# Hard AIME-style problem
test_q = "Find the number of ordered pairs of positive integers (m, n) such that m^2 * n = 20^{20}. Show your counting strategy clearly."
run_stress_test(test_q)
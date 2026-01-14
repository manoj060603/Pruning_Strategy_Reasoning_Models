import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress other non-critical warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 1024 
TEMPERATURE = 0.6

# The specific probe from the "Think Clearly" paper
PROBE_TEMPLATE = " Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"

# ==========================================
# 1. SETUP
# ==========================================
print("--- Hour 4: The Forked Attention Probe (Fixed) ---")
print(f"Loading Model: {MODEL_ID}...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- THE FIX IS HERE ---
# We force 'eager' attention to ensure outputs.attentions is not None
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    device_map="auto",
    attn_implementation="eager" 
)

probe_tokens = tokenizer.encode(PROBE_TEMPLATE, add_special_tokens=False)
probe_tensor_template = torch.tensor([probe_tokens], dtype=torch.long)

print("Model loaded with Eager Attention (Ready to Probe).")

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
# 3. THE PROBE FUNCTION
# ==========================================
def measure_attention_stateless(current_input_ids):
    """
    Forks the current context, adds the probe, runs a fresh forward pass,
    and returns the attention scores of the history.
    """
    # 1. Prepare Fork
    probe_part = probe_tensor_template.to(current_input_ids.device)
    forked_input_ids = torch.cat([current_input_ids, probe_part], dim=1)
    
    # 2. Forward Pass (Stateless)
    with torch.no_grad():
        outputs = model(
            forked_input_ids, 
            use_cache=False, 
            output_attentions=True
        )
    
    # 3. Extract Attention
    if outputs.attentions is None:
        raise ValueError("Attention weights are None. Ensure attn_implementation='eager' is set.")

    # Last layer, Average Heads -> [Batch, Seq_Len, Seq_Len]
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1)
    
    # Attention from LAST token (Probe end) to History
    # Batch 0, Row -1
    final_token_attn = avg_attn[0, -1, :]
    
    # 4. Slice to original history length
    original_length = current_input_ids.shape[1]
    history_scores = final_token_attn[:original_length]
    
    return history_scores

# ==========================================
# 4. GENERATION LOOP
# ==========================================
def run_probing_experiment(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*50}")
    print("Streaming with 'Think Clearly' Probes...\n")
    
    messages = [{"role": "user", "content": prompt_text}]
    text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    
    input_ids = inputs.input_ids
    past_key_values = None
    step_count = 0
    
    print("Generating: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Standard Step
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Update History
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            
            # CHECK & PROBE
            if is_step_boundary(next_token_id):
                step_count += 1
                
                # Run Probe
                attn_scores = measure_attention_stateless(input_ids)
                
                avg_score = attn_scores.mean().item()
                # Print marker
                print(f" [PROBE {step_count}: AvgAttn={avg_score:.5f}] ", end="", flush=True)
            
            if next_token_id == tokenizer.eos_token_id:
                print("\n\n[EOS Reached]")
                break

# ==========================================
# 5. EXECUTE
# ==========================================
test_question = "If x + 1/x = 3, find the value of x^5 + 1/x^5."
run_probing_experiment(test_question)
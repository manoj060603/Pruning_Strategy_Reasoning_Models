import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 512  # Set limit for this test run
TEMPERATURE = 0.6
TOP_P = 0.95

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
print(f"--- Hour 3: Custom Generation Loop Skeleton ---")
print(f"Loading Model: {MODEL_ID}...")

# Use bfloat16 for modern GPUs (A100/3090/4090), float16 otherwise
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully.")

# ==========================================
# 2. STEP DETECTION SETUP (From Hour 2)
# ==========================================
# We re-define this here to make the script standalone
SPLIT_WORDS = [
    "Wait", "Alternatively", "Therefore", "So", "Thus", 
    "However", "But", "Let", "First", "Next", "Finally", 
    "In conclusion", "Consequently", "Hence", "Observe",
    "Notice", "Recall", "Consider", "Assume", "Suppose",
    "Now", "We"
]

print("Mapping Step Boundary IDs...")
split_token_ids = set()
for word in SPLIT_WORDS:
    # 1. The word itself (e.g., at start of sentence)
    ids_raw = tokenizer.encode(word, add_special_tokens=False)
    if ids_raw: split_token_ids.add(ids_raw[0])
    
    # 2. The word with a preceding space (common in flow)
    ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
    if ids_space: split_token_ids.add(ids_space[0])

# 3. Newline is a strong signal for math steps
ids_nl = tokenizer.encode("\n", add_special_tokens=False)
if ids_nl: split_token_ids.add(ids_nl[-1])

def is_step_boundary(token_id):
    return token_id in split_token_ids

print(f"Mapped {len(split_token_ids)} boundary token IDs.")

# ==========================================
# 3. THE CUSTOM GENERATION LOOP
# ==========================================
def run_manual_generation(prompt_text):
    print(f"\nPrompt: {prompt_text}")
    print(f"{'='*40}")
    
    # 1. Prepare Initial Inputs
    messages = [{"role": "user", "content": prompt_text}]
    # We use chat template but do NOT tokenize yet, so we can control tensors
    text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # 2. Initialize Loop State
    past_key_values = None  # The KV Cache
    current_tokens = []     # Store full history
    chunk_count = 0
    
    print("Streaming Output:\n")
    
    # 3. The Loop
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            # A. Prepare Model Inputs
            # If we have history (past_key_values), we ONLY feed the new token
            if past_key_values is None:
                model_inputs = input_ids
            else:
                # Use the last generated token
                model_inputs = input_ids[:, -1:]

            # B. Forward Pass
            outputs = model(
                model_inputs,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # C. Update State
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # D. Sampling Strategy
            # Apply Temperature
            if TEMPERATURE > 0:
                scaled_logits = next_token_logits / TEMPERATURE
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_tensor = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (Temperature 0)
                next_token_tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            next_token_id = next_token_tensor.item()
            
            # E. Update History
            current_tokens.append(next_token_id)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # F. Decode and Print
            word = tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
            # G. CHECK BOUNDARY
            # This is the integration point for Hour 2 logic
            if is_step_boundary(next_token_id):
                chunk_count += 1
                # We use a visual marker for now
                print(f" [CHUNK {chunk_count} FINISHED] ", end="", flush=True)
            
            # H. Stop Condition
            if next_token_id == tokenizer.eos_token_id:
                print("\n\n[EOS Reached]")
                break
                
    return tokenizer.decode(current_tokens)

# ==========================================
# 4. EXECUTION
# ==========================================
# Test with a math problem that requires steps
test_question = "Calculate the sum of the first 5 prime numbers and explain your steps."
run_manual_generation(test_question)
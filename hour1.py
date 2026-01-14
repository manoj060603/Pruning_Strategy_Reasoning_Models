import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import time
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_ID = "HuggingFaceH4/aime_2024" # Standard AIME dataset source
SUBSET_SIZE = 10  # Number of questions to run for the baseline
MAX_NEW_TOKENS = 8192 # Reasoning models need high limits (often ~4k-8k)

# ==========================================
# 1. SETUP & LOADING
# ==========================================
print(f"--- Starting Hour 1: Setup & Baseline ---")
print(f"Loading Model: {MODEL_ID}...")

# Determine device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 for Ampere+ GPUs (A100, 3090, 4090), else float16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model with device_map="auto" to handle VRAM allocation automatically
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully.")

# Load Dataset
print(f"Loading Dataset: {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split="train") # AIME usually comes in train/test splits
# Select just the first N examples for the baseline
subset = dataset.select(range(SUBSET_SIZE))
print(f"Loaded {len(subset)} examples for baseline testing.")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_answer(text):
    """
    Extracts the answer from the \\boxed{} latex command.
    Standard for Math datasets.
    """
    # Look for \boxed{...}
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches:
        # Return the last match (models usually box the final answer at the end)
        return matches[-1].strip()
    return None

def normalize_answer(answer_str):
    """
    Normalizes answers for comparison (e.g., '005' -> '5').
    """
    if answer_str is None:
        return None
    try:
        # Remove commas, spaces, dollar signs
        clean = re.sub(r"[,\s\$]", "", answer_str)
        # Try converting to float/int for numerical comparison
        return str(float(clean))
    except:
        return answer_str.strip()

# ==========================================
# 3. BASELINE EXECUTION LOOP
# ==========================================
results = []
total_start_time = time.time()

print("\n--- Starting Generation Loop ---")

for i, example in enumerate(subset):
    question = example['problem']
    ground_truth = example['solution'] # Use 'answer' or 'solution' depending on dataset version
    
    # Extract just the boxed part of ground truth for comparison
    gt_clean = extract_answer(ground_truth) 
    if gt_clean is None: gt_clean = ground_truth # Fallback if no box in GT
    
    print(f"\nProcessing Q{i+1}/{SUBSET_SIZE}...")
    
    # Format Prompt (Chat Template)
    messages = [
        {"role": "user", "content": question}
    ]
    prompt_inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    input_len = prompt_inputs.shape[1]
    
    # Generate (Standard / No Pruning)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            prompt_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,      # DeepSeek-R1 usually recommends greedy or low temp, but we use defaults
            temperature=0.6,     # Slightly creative for reasoning
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    # Decoding
    output_ids = outputs[0][input_len:] # Slice off the prompt
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Metrics
    token_count = len(output_ids)
    time_taken = end_time - start_time
    tokens_per_sec = token_count / time_taken
    
    # Evaluation
    model_ans = extract_answer(generated_text)
    is_correct = normalize_answer(model_ans) == normalize_answer(gt_clean)
    
    results.append({
        "id": i,
        "correct": is_correct,
        "token_count": token_count,
        "time": time_taken,
        "tps": tokens_per_sec
    })
    
    print(f"  > Tokens: {token_count}")
    print(f"  > Time: {time_taken:.2f}s ({tokens_per_sec:.2f} tok/s)")
    print(f"  > Answer Found: {model_ans} | GT: {gt_clean}")
    print(f"  > Correct: {is_correct}")

# ==========================================
# 4. FINAL REPORT
# ==========================================
accuracy = sum([r['correct'] for r in results]) / len(results) * 100
avg_tokens = np.mean([r['token_count'] for r in results])
avg_tps = np.mean([r['tps'] for r in results])

print("\n" + "="*40)
print("HOUR 1: BASELINE RESULTS")
print("="*40)
print(f"Model: {MODEL_ID}")
print(f"Dataset: AIME 2024 (First {SUBSET_SIZE} samples)")
print("-" * 40)
print(f"Accuracy:        {accuracy:.2f}%")
print(f"Avg Token Count: {avg_tokens:.1f}")
print(f"Avg Speed:       {avg_tps:.2f} tok/s")
print("="*40)

# Save baseline metrics for later comparison
import json
with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Baseline results saved to 'baseline_results.json'")
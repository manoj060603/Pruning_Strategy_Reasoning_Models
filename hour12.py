import json
import re
import numpy as np
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
# File paths generated in previous hours
FILES = {
    "Baseline (Hour 1)": "baseline_results.json",
    "Exp A (Conservative)": "results_exp_a_windowed.json",
    "Exp B (Balanced)": "results_exp_b_balanced.json",
    "Exp C (Aggressive)": "results_exp_c_aggressive.json"
}

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def robust_extract_answer(text):
    """
    Refined Regex logic to extract AIME answers.
    Prioritizes \boxed{}, then \framebox{}, then looks for "answer is X".
    """
    if not text: return None
    
    # 1. Standard LaTeX Box (The Gold Standard)
    # Matches \boxed{123} or \boxed{ 123 }
    box_match = re.findall(r"\\boxed\s*\{([^}]+)\}", text)
    if box_match:
        return box_match[-1].strip()
    
    # 2. Alternative Boxes
    # Matches \framebox{123} or \textbf{123}
    alt_box = re.findall(r"\\(?:framebox|textbf)\s*\{([^}]+)\}", text)
    if alt_box:
        return alt_box[-1].strip()
        
    # 3. Fallback: "The answer is..." patterns (Common in DeepSeek)
    text_lower = text.lower()
    # Look for number at the very end or after specific phrases
    fallback = re.findall(r"(?:answer is|result is|value is)\s*[:\s]*([0-9\.]+)", text_lower)
    if fallback:
        return fallback[-1]
        
    return None

def normalize_answer(ans):
    """Normalizes string answers for comparison (removes spaces, commas, $)."""
    if not ans: return ""
    try:
        # Remove $, commas, spaces
        clean = re.sub(r"[,\s\$]", "", str(ans))
        # Convert to float to handle 10 vs 10.0, then back to string to handle string compare
        return str(float(clean))
    except:
        return str(ans).strip()

def check_correctness(model_ans, gt_ans):
    return normalize_answer(model_ans) == normalize_answer(gt_ans)

# ==========================================
# 2. DATA PROCESSING LOOP
# ==========================================
print("--- Hour 12: Data Parsing & Compilation ---")

final_stats = []

for experiment_name, filename in FILES.items():
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping...")
        continue
        
    with open(filename, "r") as f:
        data = json.load(f)
        
    print(f"Processing {experiment_name} ({len(data)} samples)...")
    
    # Metrics containers
    correct_count = 0
    total_tokens_used = []
    total_tokens_virtual = [] # What it WOULD have been without pruning
    pruned_counts = []
    
    for entry in data:
        # 1. Handle different field names between Baseline (Hour 1) and Experiments
        # Hour 1 used "token_count", Experiments used "tokens_total" + "tokens_pruned"
        if "token_count" in entry:
            # Baseline Logic
            generated = entry["token_count"]
            pruned = 0
            final_text = entry.get("model_ans", "") # Hour 1 stored answer, not full text usually
            gt = entry.get("gt_answer", "") # We might need to fetch from dataset if not saved, but Hour 1 saved 'gt_clean' logic
            
            # Re-verify logic not possible if full text missing in Hour 1 log, 
            # so we trust the 'correct' flag from JSON
            is_correct = entry["correct"]
            
        else:
            # Experiment Logic
            generated = entry["tokens_total"]
            pruned = entry["tokens_pruned"]
            # Re-run extraction to be rigorous
            # (In a real run, we'd use the full text, here we trust the log's answer field 
            # if full text isn't saved to save space, but let's assume we use the saved fields)
            is_correct = entry["correct"]
            
        # 2. Accumulate
        total_tokens_used.append(generated)
        total_tokens_virtual.append(generated + pruned)
        pruned_counts.append(pruned)
        if is_correct:
            correct_count += 1
            
    # 3. Compute Aggregates
    n = len(data)
    if n == 0: continue
    
    acc = (correct_count / n) * 100
    avg_tokens_used = np.mean(total_tokens_used)
    avg_tokens_virtual = np.mean(total_tokens_virtual)
    avg_pruned = np.mean(pruned_counts)
    
    # Calculate Reduction % 
    # Formula: (Virtual - Actual) / Virtual
    if avg_tokens_virtual > 0:
        reduction_pct = (avg_pruned / avg_tokens_virtual) * 100
    else:
        reduction_pct = 0
        
    final_stats.append({
        "Experiment": experiment_name,
        "Accuracy": acc,
        "Avg Tokens (Actual)": avg_tokens_used,
        "Avg Tokens (Pruned)": avg_pruned,
        "Reduction %": reduction_pct
    })

# ==========================================
# 3. GENERATE COMPARISON TABLE
# ==========================================
df = pd.DataFrame(final_stats)

# Format for display
pd.options.display.float_format = '{:.2f}'.format

print("\n" + "="*60)
print("FINAL PROJECT RESULTS: AIME 2024 PRUNING ANALYSIS")
print("="*60)
print(df.to_markdown(index=False))
print("="*60)

# ==========================================
# 4. EXPORT
# ==========================================
csv_filename = "final_project_comparison.csv"
df.to_csv(csv_filename, index=False)
print(f"\nSuccess. Final data exported to {csv_filename}")
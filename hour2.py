import torch
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
# We use the tokenizer from the exact model to ensure ID mismatch doesn't happen
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ==========================================
# 1. SETUP
# ==========================================
print(f"--- Starting Hour 2: Robust Segmentation Setup ---")
print(f"Loading Tokenizer from: {MODEL_ID}...")

# Load tokenizer
# valid_token ensures we don't crash on slightly malformed inputs
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Tokenizer loaded.")

# ==========================================
# 2. DEFINE SPLIT WORDS
# ==========================================
# These are the transition markers defined in the "Think Clearly" paper (Appendix A.2)
# They signal the start of a new reasoning chunk.
SPLIT_WORDS = [
    "Wait", "Alternatively", "Therefore", "So", "Thus", 
    "However", "But", "Let", "First", "Next", "Finally", 
    "In conclusion", "Consequently", "Hence", "Observe",
    "Notice", "Recall", "Consider", "Assume", "Suppose",
    "Now", "We" # "We" and "Now" are also common in math proofs
]

# ==========================================
# 3. DYNAMIC ID MAPPING
# ==========================================
print(f"Mapping {len(SPLIT_WORDS)} keywords to Token IDs...")

split_token_ids = set()

for word in SPLIT_WORDS:
    # Case A: Word at start of sentence (e.g., "Wait...")
    # add_special_tokens=False prevents adding <|im_start|> or similar
    ids_raw = tokenizer.encode(word, add_special_tokens=False)
    
    # Case B: Word in middle of sentence (e.g., "... value. Wait...")
    # Most tokenizers treat " Wait" (with space) as a distinct token ID
    ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
    
    # We trigger on the FIRST token of the word.
    # Even if "Alternatively" splits into multiple tokens, catching the first one is enough.
    if ids_raw:
        split_token_ids.add(ids_raw[0])
    if ids_space:
        split_token_ids.add(ids_space[0])

# Case C: Newline characters often delimit steps in math
# We grab the ID for '\n'
newline_ids = tokenizer.encode("\n", add_special_tokens=False)
if newline_ids:
    split_token_ids.add(newline_ids[-1])

print(f"Success. Mapped {len(split_token_ids)} unique Token IDs that act as Step Boundaries.")

# ==========================================
# 4. THE DETECTION FUNCTION
# ==========================================
def is_step_boundary(token_id):
    """
    Checks if a given token_id marks the start of a new reasoning step.
    This is an O(1) lookup.
    """
    return token_id in split_token_ids

# ==========================================
# 5. VERIFICATION TEST
# ==========================================
print("\n--- Running Verification Test ---")

# A sample string mimicking a reasoning trace
# We expect boundaries at: "Wait", "So", "Thus", and "\n"
test_text = "x is 5.\nWait, let me double check. So, if we integrate... Thus, the answer is 10."

print(f"Test Input: '{test_text}'")

# Encode the test string
test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
print(f"Token Sequence: {test_tokens}")

print("\nScanning sequence for boundaries...")
found_count = 0

for i, t_id in enumerate(test_tokens):
    # Decode strictly for display purposes
    token_str = tokenizer.decode([t_id])
    
    if is_step_boundary(t_id):
        print(f"  [BOUNDARY FOUND] at Index {i}: ID={t_id} | Token='{token_str}'")
        found_count += 1
    else:
        # Optional: Print non-boundary tokens to see flow
        # print(f"  Index {i}: {token_str}")
        pass

# ==========================================
# 6. RESULTS
# ==========================================
if found_count > 0:
    print(f"\nSUCCESS: Detected {found_count} boundaries.")
    print("You can now safely use 'is_step_boundary(token_id)' in your generation loop.")
else:
    print("\nWARNING: No boundaries detected. Check the SPLIT_WORDS list or Tokenizer.")
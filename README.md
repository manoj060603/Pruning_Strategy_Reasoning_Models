# ClearThink: Adaptive Attention-Based Pruning for Reasoning Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Model](https://img.shields.io/badge/Model-DeepSeek--R1--Distill-gray)

**ClearThink** is an inference-time optimization engine that reduces the computational cost of "System 2" reasoning models by surgically removing redundant Chain-of-Thought (CoT) tokens.

This project implements concepts from **[Think Clearly (Choi et al., 2025)](https://arxiv.org/abs/2507.08806)** and **[TRAAC (Singh et al., 2025)](https://arxiv.org/abs/2510.01581)**, specifically focusing on **Attention-Based Step Pruning** to mitigate "Overthinking" while maintaining accuracy on complex math benchmarks like **AIME 2024**.

---

## 🚀 Key Features

*   **Step-Aware Segmentation:** Automatically detects reasoning boundaries (e.g., "Wait," "Therefore," "Alternatively") using token-ID mapping rather than brittle string matching.
*   **Forked Attention Probing:** Uses a stateless "Time is up" probe to measure the global importance of past reasoning chunks without polluting the KV cache.
*   **Context Refreshing:** A novel pruning implementation that flushes the KV cache and re-tokenizes inputs to maintain **Rotary Positional Embedding (RoPE)** coherence after deletion.
*   **Adaptive Thresholds:** Supports Conservative (Hard Mode), Balanced, and Aggressive pruning strategies to test the trade-off between efficiency and accuracy.

---

## 📊 Results (AIME 2024)

We evaluated the pipeline on a subset of the AIME 2024 dataset using `DeepSeek-R1-Distill-Qwen-7B`.

| Strategy | Pruning Ratio | Accuracy | Avg Token Reduction | Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | N/A | **N/A** | 0% | Often hits context limits (8k) on hard problems. |
| **Conservative** | Bottom 25% | **40.0%** | ~20% | **Optimal.** Removes arithmetic scaffolding; solves longest problems. |
| **Balanced** | Bottom 40% | 30.0% | ~21% | Diminishing returns. Removed too much context. |
| **Aggressive** | Bottom 60% | 20.0% | ~22% | Failure mode. Model enters "Regeneration Loops" (Underthinking). |

**Key Finding:** Hard math problems require a "Conservative" pruning approach. Aggressive pruning causes the model to hallucinate or regenerate deleted steps, negating efficiency gains.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ClearThink.git
    cd ClearThink
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch transformers accelerate datasets numpy pandas
    ```

3.  **Hardware Requirements:**
    *   GPU with 24GB+ VRAM (RTX 3090/4090 or A100/A10G).
    *   Implementation uses `bfloat16` for memory efficiency.

---

## 📂 Project Structure

```text
ClearThink/
├── src/
│   ├── baseline.py           # (Hour 1) Standard generation benchmark
│   ├── robust_probe.py       # (Hour 4) Stateless attention measurement
│   ├── context_refresh.py    # (Hour 6) The core pruning logic
│   └── run_experiment.py     # (Hour 9-11) Main execution script
├── results/
│   ├── results_exp_a.json    # Logs for Conservative Mode
│   └── results_exp_b.json    # Logs for Balanced Mode
├── analysis.py               # (Hour 12) Data processing & Table generation
└── README.md
```

---

## 🧠 Methodology

### 1. The "Probe"
To determine if a reasoning step is useful, we pause generation at step boundaries and inject a specific prompt:
> *"Time is up. Given the time I've spent... summarize in one sentence."*

We run a forward pass on this sequence and extract the attention weights from the `</think>` token to all previous reasoning chunks.

### 2. The "Context Refresh" (The Fix for RoPE)
Simply slicing the Key-Value (KV) cache tensor breaks modern LLMs because of Rotary Positional Embeddings. This project solves this by:
1.  Identifying low-score text chunks.
2.  Reconstructing the prompt string *without* those chunks.
3.  **Flushing the KV Cache (`None`)** and re-feeding the optimized prompt.
4.  This forces the model to re-compute correct positional embeddings for the shortened sequence.

---

## 💻 Usage

**1. Run the Baseline:**
```bash
python src/baseline.py
```

**2. Run Experiments:**
You can modify the `PRUNING_RATIO` in `run_experiment.py` (0.25, 0.40, 0.65).
```bash
python src/run_experiment.py
```

**3. Generate Comparison Table:**
```bash
python analysis.py
```

---

## 🔮 Future Work

Based on the findings, the next iteration of this project will implement **Confidence-Weighted Attention**. 

By calculating the **Harmonic Mean** of token probabilities within a chunk, we can distinguish between:
*   **Redundant Fluff:** High Confidence + Low Attention ($\rightarrow$ Prune)
*   **Critical Leaps:** Low Confidence + Low Attention ($\rightarrow$ Protect)

This acts as a zero-shot difficulty detector to prevent the "Underthinking" observed in Aggressive Mode.

---

## 📜 References

1.  **Think Clearly: Improving Reasoning via Redundant Token Pruning**
    *Choi et al. (2025)* - [arXiv:2507.08806](https://arxiv.org/abs/2507.08806)
2.  **Think Right: Learning to Mitigate Under-Over Thinking (TRAAC)**
    *Singh et al. (2025)* - [arXiv:2510.01581](https://arxiv.org/abs/2510.01581)

---

## 📄 License

MIT License. Feel free to use this code for research and educational purposes.

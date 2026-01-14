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

## 📜 References

1.  **Think Clearly: Improving Reasoning via Redundant Token Pruning**
    *Choi et al. (2025)* - [arXiv:2507.08806](https://arxiv.org/abs/2507.08806)
2.  **Think Right: Learning to Mitigate Under-Over Thinking (TRAAC)**
    *Singh et al. (2025)* - [arXiv:2510.01581](https://arxiv.org/abs/2510.01581)

---

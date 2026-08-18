# svd-lowrank-demo

A from-scratch study of **Singular Value Decomposition (SVD)** and how it connects to low-rank matrix compression — the mathematical foundation behind LoRA, model pruning, and weight compression in deep learning.

Same approach as `mini-llm-api` and `lora-from-paper`: read the theory, implement it in code, document what you learn. Project is ongoing.

---

## What This Explores

Any weight matrix `W` in a neural network can be decomposed as:

```
W = U · Σ · Vᵀ
```

- `U` — left singular vectors (output constructors)
- `Σ` — diagonal matrix of singular values (importance scores, sorted descending)
- `Vᵀ` — right singular vectors (pattern extractors / feature detectors)

By keeping only the top `k` singular values (truncated SVD), you get a compressed approximation `W_k` that uses far fewer parameters while preserving the most important structure of the transformation.

---

## Project Structure

| Path | Contents |
| :--- | :--- |
| `src/svd_compression.py` | Core functions: `compute_svd`, `truncate_svd`, `reconstruct_matrix`, `compute_reconstruction_error` |
| `src/compress_model.py` | Applies SVD compression to a neural network layer |
| `notebooks/` | Experiments and visualizations (to be added) |
| `math-notes/` | Derivations and shape analysis |
| `experiments/` | Compression results at different ranks |

---

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install numpy torch

python src/svd_compression.py
```

---

## Study Notes
See [NOTES.md](NOTES.md) for the full conceptual breakdown: what `U`, `Σ`, `Vᵀ` mean in ML terms, why singular values are sorted, how truncation forces the matrix shapes to change, and how reconstruction error is measured.

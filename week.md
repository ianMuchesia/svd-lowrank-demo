Alright. **Cycle 9 — Week 3 only. Strictly in scope. No LoRA experiments. No memory profiling.**

---

# CYCLE 9 — WEEK 3 (27 June – 3 July 2026)

## Project: `svd-lowrank-demo`

## Focus: SVD Weight Compression + Compression–Accuracy Tradeoff

## Core Goal

This week is about answering one question:

> How much of a neural network's weight matrix is actually necessary?

You will compress trained model weights using **Singular Value Decomposition (SVD)** and measure what happens to:

* Accuracy
* Model size
* Inference speed

This is your first exposure to **post-training model compression**.

---

# Final Repository Structure

```text
svd-lowrank-demo/
│
├── src/
│   ├── svd_compression.py
│   ├── compress_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── compression_analysis.ipynb
│
├── experiments/
│   ├── singular_values.png
│   ├── compression_tradeoff.png
│   ├── rank_25_results.json
│   ├── rank_50_results.json
│   ├── rank_100_results.json
│   └── compression_summary.md
│
├── math-notes/
│   └── svd_derivation.md
│
├── checkpoints/
│   └── trained_model.pt
│
└── README.md
```

---

# What You Are Building Conceptually

Suppose a trained layer has:

```text
W ∈ R(512 × 512)
```

Total parameters:

```text
512 × 512

= 262,144
```

---

SVD decomposes:

```text
W = UΣVᵀ
```

where:

```text
U = left singular vectors

Σ = singular values

Vᵀ = right singular vectors
```

---

Instead of keeping everything:

```text
σ1 σ2 σ3 σ4 σ5 ...
```

keep only:

```text
σ1 σ2 σ3
```

and discard the rest.

---

Approximation:

```text
W ≈ Uk Σk Vkᵀ
```

This creates a much smaller representation.

---

# WEEK 3 — DAY BY DAY

---

# Day 1 — Implement SVD Compression

## File

```text
src/svd_compression.py
```

Implement:

```python
compress_matrix(weight_matrix, rank)
```

---

Workflow:

```text
Weight Matrix
      ↓
SVD
      ↓
U Σ Vᵀ
      ↓
Keep top-k singular values
      ↓
Reconstruct matrix
      ↓
Compressed Weight
```

---

Functions Required

```python
compute_svd()

truncate_svd()

reconstruct_matrix()

compression_ratio()
```

---

Verification

Measure reconstruction error:

```text
||W - Wcompressed||
```

Smaller is better.

---

# Day 2 — Analyze Singular Value Spectrum

## Notebook

```text
notebooks/compression_analysis.ipynb
```

Load trained layer.

Compute:

```python
U,S,Vt = np.linalg.svd(W)
```

Plot:

```text
Singular Value
vs
Index
```

Save:

```text
experiments/singular_values.png
```

---

Observation Goal

Determine:

```text
How many singular values
contain most information?
```

Typical result:

```text
First 20-50 values dominate.
```

---

# Day 3 — Rank 25 Compression

Compress layer using:

```text
k = 25
```

---

Evaluate:

```text
Accuracy
Inference Time
Model Size
```

Save:

```text
experiments/rank_25_results.json
```

---

Calculate:

```text
Compression Ratio
```

Example:

```text
Original:
262,144 params

Compressed:
25,600 params

≈ 10x smaller
```

---

# Day 4 — Rank 50 Compression

Repeat:

```text
k = 50
```

Save:

```text
experiments/rank_50_results.json
```

---

Compare:

```text
Rank 25
vs
Rank 50
```

Expected:

```text
Better accuracy

Lower compression
```

---

# Day 5 — Rank 100 Compression

Repeat:

```text
k = 100
```

Save:

```text
experiments/rank_100_results.json
```

Expected:

```text
Near-original accuracy

Less compression
```

---

# Day 6 — Layer-wise Compression Experiments

Not all layers compress equally.

Test:

### Strategy A

```text
Compress all layers equally
```

---

### Strategy B

```text
Compress only dense layers
```

---

### Strategy C

```text
Aggressive compression early layers

Light compression later layers
```

---

Measure:

```text
Accuracy
Size
Speed
```

Document findings.

---

# Day 7 — Tradeoff Analysis

## File

```text
experiments/compression_summary.md
```

Create table:

| Rank     | Accuracy | Model Size | Compression Ratio |
| -------- | -------- | ---------- | ----------------- |
| Original | ?        | ?          | 1x                |
| 25       | ?        | ?          | ?                 |
| 50       | ?        | ?          | ?                 |
| 100      | ?        | ?          | ?                 |

---

Create plot:

```text
experiments/compression_tradeoff.png
```

X-axis:

```text
Compression Ratio
```

Y-axis:

```text
Accuracy
```

---

Goal:

Visualize where performance starts collapsing.

---

# Required Math Notes

## File

```text
math-notes/svd_derivation.md
```

---

# Singular Value Decomposition

Definition:

```text
W = UΣVᵀ
```

where:

```text
UᵀU = I

VᵀV = I
```

Orthogonal matrices.

---

# Low-Rank Approximation

Keep only:

```text
Top k singular values
```

Then:

```text
Wk = Uk Σk Vkᵀ
```

---

# Reconstruction Error

Measure:

```text
||W - Wk||F
```

Frobenius norm.

---

Interpretation:

```text
Smaller error

↓

Better approximation

↓

Higher retained accuracy
```

---

# Parameter Count

Original:

```text
m × n
```

Compressed:

```text
mk + k + nk
```

---

Example

```text
512 × 512

Original:
262,144
```

---

Rank 50:

```text
512×50
+
50
+
512×50

=

51,250
```

---

Compression:

```text
262,144 / 51,250

≈ 5.1x
```

---

# Deliverables Required This Week

## Source Code

```text
src/svd_compression.py
src/compress_model.py
src/evaluate.py
src/utils.py
```

---

## Notebook

```text
notebooks/compression_analysis.ipynb
```

---

## Experiments

```text
experiments/singular_values.png

experiments/compression_tradeoff.png

experiments/rank_25_results.json

experiments/rank_50_results.json

experiments/rank_100_results.json

experiments/compression_summary.md
```

---

## Math Notes

```text
math-notes/svd_derivation.md
```

---

# Completion Checklist

Week 3 is complete when:

* SVD implemented from scratch
* Singular value spectrum visualized
* Rank 25 experiment completed
* Rank 50 experiment completed
* Rank 100 experiment completed
* Compression vs accuracy plot created
* Reconstruction error measured
* Compression tradeoff documented

---

# What You Achieve This Week

You move from:

```text
"I know SVD from linear algebra."
```

to:

```text
"I can compress trained neural networks
and quantify the cost of compression."
```

This is one of the foundational techniques behind model optimization, deployment, and efficient inference.

---

### Next Week Preview (Cycle 9 Week 4)

You will build a full **memory profiling framework** and compare:

```text
Vanilla Transformer
vs
LoRA
vs
Gradient Checkpointing
vs
Mixed Precision (FP16)
```

to understand exactly where training memory is consumed and how modern systems reduce it.

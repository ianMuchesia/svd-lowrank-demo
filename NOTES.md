# Notes: SVD and Low-Rank Compression

## 1. What the Weight Matrix W Actually Is
In a standard layer, input `x` (e.g. a 512-dim embedding) is multiplied by `W` to produce output `y`:
```
y = Wx
```
`W` is a `512 × 512` transformation matrix — it contains all the learned logic to read the input, detect patterns in it, and assemble a new output for the next layer.

## 2. SVD: Breaking Down the Transformation
SVD proves that any matrix `W` can be perfectly decomposed into three sequential operations:
```
W = U · Σ · Vᵀ
```
So instead of `y = Wx`, we compute `y = (UΣVᵀ)x` — the data passes through `Vᵀ`, then `Σ`, then `U`.

**What each matrix means in ML terms:**

- **`Vᵀ` (Right Singular Vectors) — The Pattern Extractors**
  - `Vᵀ` is a `512 × 512` orthogonal matrix.
  - Each row is a "feature detector" or "concept" the model learned.
  - Multiplying `x` by `Vᵀ` measures how much of each of the 512 concepts is present in the input — it re-expresses the input in the model's coordinate system.

- **`Σ` (Singular Values) — The Importance Scores**
  - A `512 × 512` diagonal matrix — all zeros except for 512 numbers on the diagonal.
  - Each singular value corresponds to one feature detector in `Vᵀ`. Large value = that pattern is critical. Tiny value = mostly noise.
  - **Always sorted descending:** `σ₁ ≥ σ₂ ≥ ... ≥ σ₅₁₂ ≥ 0`. This is a mathematical guarantee from the SVD algorithm — it finds directions of maximum variance in order. This sorting is what makes compression possible.

- **`U` (Left Singular Vectors) — The Output Constructors**
  - A `512 × 512` orthogonal matrix.
  - Each column is a "base output" — a fundamental building block for the output space.
  - After `Σ` amplifies important signals and mutes weak ones, `U` maps them back into the 512-dim space the next layer expects.

## 3. Truncation — Where the Shapes Come From
With exact SVD, all three matrices are `512 × 512`. But since singular values are sorted, the bottom values are essentially useless.

Setting compression rank `k = 50` means: keep the top 50 patterns, delete the rest 462.

**How truncation changes the shapes:**
- `Σ`: delete bottom 462 rows and rightmost 462 columns → `50 × 50`
- `Vᵀ`: the 462 deleted feature detectors have no importance score to connect to — delete those rows → `50 × 512`
- `U`: the 462 deleted output constructors multiply against nothing — delete those columns → `512 × 50`

## 4. Reconstruction
Multiply the three truncated matrices back together:
```
W_compressed = U_k · Σ_k · Vᵀ_k
(512×50) · (50×50) · (50×512) = (512×512)
```
Inner dimensions match. Result is `512 × 512` — exact same shape as the original `W`, so it can be swapped back into the network seamlessly.

It's not a perfect copy — it was reconstructed through a bottleneck of only `k` patterns — so it has lost some precision. To measure how much:
```python
error = np.linalg.norm(W_original - W_compressed)  # Frobenius norm
```

## 5. Parameter Count — Why This Saves Memory
Original `W`: `512 × 512 = 262,144` parameters.

Truncated SVD at rank `k`:
- `U_k`: `512 × k`
- `S_k`: `k` values (diagonal only)
- `Vᵀ_k`: `k × 512`
- **Total**: `512k + k + 512k = 1025k`

At `k = 50`: `1025 × 50 = 51,250` parameters — **5× fewer** than the original.
At `k = 10`: `~10,250` — **25× fewer**.

The tradeoff: lower rank = fewer params = more reconstruction error. The singular value spectrum tells you how much quality you lose at each rank.


## 6. Why a Random Matrix Gives a Flat Singular Value Spectrum
When you plot the singular values of a randomly initialized matrix (e.g. `np.random.randn(512, 512)`), the graph looks nearly linear — no sharp drop-off, no "elbow."

The reason: a random matrix has no structure. Every direction in the input space is equally important — there are no dominant patterns and no noise. The singular values spread out evenly because all 512 "features" carry roughly the same variance.

**A trained weight matrix is the opposite.** After training, a few singular values become very large (the model concentrated its learned knowledge into a small number of dominant patterns), and the rest drop off sharply. That steep drop is what the elbow in the spectrum captures — it tells you where "signal" ends and "noise" begins.

**The practical lesson:** always run SVD analysis on a *trained* weight matrix, not a random one. Loading from a checkpoint (e.g. GPT-2's `c_attn` layer) gives you the real spectrum with a visible elbow that tells you what rank to truncate at.
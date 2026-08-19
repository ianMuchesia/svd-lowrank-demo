# LoRA vs Full Fine-Tuning

| Method | Val Accuracy | Trainable Params | Avg Epoch Time (s) | Peak GPU Memory (MB) |
|---|---|---|---|---|
| Full FT | 54.47% | 124,439,808 | 187.8 | 3171.8 |
| LoRA r=4 | 54.60% | 147,456 | 130.1 | 4541.8 |
| LoRA r=8 | 54.60% | 294,912 | 144.4 | 4543.2 |
| LoRA r=16 | 54.74% | 589,824 | 130.3 | 4549.1 |

## Notes

- All LoRA runs share the same alpha, so trainable-parameter and memory differences are attributable to rank alone.
- Full FT uses a lower learning rate than the LoRA runs (see configs) since all weights are trainable.
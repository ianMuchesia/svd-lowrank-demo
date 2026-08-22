import time
import json
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.svd_compression import (
    compute_svd,
    truncate_svd,
    reconstruct_matrix,
    compute_reconstruction_error,
    compressed_parameter_count,
    compression_ratio,
)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/Torch values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value



def evaluate_compression(weight_matrix:np.ndarray,rank:int)->dict:
    
    m,n = weight_matrix.shape
    
    original_params = m * n
    
    compressed_params = compressed_parameter_count(m,n,rank)
    ratio = compression_ratio(m,n,rank)
    
    
    
    #also report construction error, since it explains the accuracy story
    U, S,Vt = compute_svd(weight_matrix)
    
    Uk,Sk,Vtk = truncate_svd(U,S,Vt,rank)
    
    
    W_compressed = reconstruct_matrix(Uk,Sk,Vtk)
    
    recon_error = compute_reconstruction_error(weight_matrix,W_compressed)
    
    
    return {
        "rank":rank,
        "original_params":original_params,
        "compressed_params":compressed_params,
        "compressed_ratio":ratio,
        "params_saved":original_params - compressed_params,
        "reconstruction_error":recon_error
    }
    
    
    
    
    
    
    
# inference speed
@torch.no_grad()
def measure_inference_time(model,input_batch,device, n_warmup=3,n_runs=10)->dict:
    
    model.eval()
    model.to(device)
    input_batch = input_batch.to(device)
    
    # warmup (important on GPU: first calls include CUDA context / kernel compile overhead)
    for _ in range(n_warmup):
        _ = model(input_batch)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
        
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(input_batch)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        
    
    times_np = np.array(times)

    
    
    
    return {
    "mean_time_sec": float(times_np.mean()),
    "std_time_sec": float(times_np.std()),
    "min_time_sec": float(times_np.min()),
    "n_runs": n_runs,
    "batch_size": input_batch.shape[0],
    "seq_len": input_batch.shape[1],
}
        
        
        
#Accuracy / validation loss (cross-entropy)
@torch.no_grad()
def evaluate_validation_loss(model,val_dataloader:DataLoader,device)->dict:
    
    model.eval()
    
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        
        # HF models return mean loss over non-masked tokens by default.
        # To aggregate correctly across batches of different sizes, we
        # need the token count, not just the mean loss.
        n_tokens = (labels != -100).sum().item()

        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
            
            
    avg_loss = total_loss / total_tokens
    perplexity = float(np.exp(avg_loss))
        
    return {
    "cross_entropy_loss": avg_loss,
    "perplexity": perplexity,
    "total_tokens": total_tokens,
    }
    
        
        
def run_full_evaluation(
    original_model,
    svd_model,
    weight_matrix: np.ndarray,
    rank: int,
    val_dataloader: DataLoader,
    sample_input_batch: torch.Tensor,
    device,
    results_path
) -> dict:
    print(f"=== Compression (rank={rank}) ===")
    compression_results = evaluate_compression(weight_matrix, rank)
    for k, v in compression_results.items():
        print(f"  {k}: {v}")

    print("\n=== Inference speed: original ===")
    orig_speed = measure_inference_time(original_model, sample_input_batch, device)
    for k, v in orig_speed.items():
        print(f"  {k}: {v}")

    print("\n=== Inference speed: SVD ===")
    svd_speed = measure_inference_time(svd_model, sample_input_batch, device)
    for k, v in svd_speed.items():
        print(f"  {k}: {v}")

    print("\n=== Validation loss: original ===")
    orig_loss = evaluate_validation_loss(original_model, val_dataloader, device)
    for k, v in orig_loss.items():
        print(f"  {k}: {v}")

    print("\n=== Validation loss: SVD ===")
    svd_loss = evaluate_validation_loss(svd_model, val_dataloader, device)
    for k, v in svd_loss.items():
        print(f"  {k}: {v}")

    speedup = orig_speed["mean_time_sec"] / svd_speed["mean_time_sec"]
    loss_delta = svd_loss["cross_entropy_loss"] - orig_loss["cross_entropy_loss"]

    print(f"\n=== Summary ===")
    print(f"  Compression ratio: {compression_results['compressed_ratio']:.2f}x fewer params")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Loss delta (SVD - original): {loss_delta:+.4f}")

    summary = {
        "compression": compression_results,
        "speed_original": orig_speed,
        "speed_svd": svd_speed,
        "loss_original": orig_loss,
        "loss_svd": svd_loss,
        "speedup": speedup,
        "loss_delta": loss_delta,
        "Compression_ratio": f"{compression_results['compressed_ratio']:.2f}x fewer params"
    }
    summary_json = _to_jsonable(summary)
    if not isinstance(summary_json, dict):
        raise TypeError("Serialized summary must be a dictionary")
    
    with open(results_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    print(f"[{rank}] Saved results to {results_path}")
    
    return summary_json
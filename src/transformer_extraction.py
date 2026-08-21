import torch
import torch.nn as nn
import numpy as np
from src.svd_compression import compute_svd, truncate_svd

def extract_layers(model):
    for name, module in model.named_modules():
        print(name)
        
        

def create_svd_linear(original_layer, rank, device):
    """
    original_layer: an HF Conv1D module (used for c_fc / c_proj in GPT-2).
    Conv1D.weight has shape (in_features, out_features); forward is x @ W + b.
    """
    W = original_layer.weight.data.detach().cpu().numpy()   # shape (in, out)
    bias = original_layer.bias

    in_features, out_features = W.shape

    U, S, Vt = compute_svd(W)
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, rank)

    # We want: x @ W ≈ x @ (Uk*Sk) @ Vtk = (x @ (Uk*Sk)) @ Vtk
    # nn.Linear computes y = x @ weight.T + b, so:
    #   layer1.weight (rank, in_features)  = (Uk * Sk).T
    #   layer2.weight (out_features, rank) = Vtk.T
    svd_layer = nn.Sequential(
        nn.Linear(in_features, rank, bias=False),
        nn.Linear(rank, out_features, bias=(bias is not None)),
    )

    layer1_weight = (Uk * Sk).T          # (rank, in_features)
    layer2_weight = Vtk.T                # (out_features, rank)

    svd_layer[0].weight.data = torch.from_numpy(layer1_weight).float().to(device)
    svd_layer[1].weight.data = torch.from_numpy(layer2_weight).float().to(device)

    if bias is not None:
        svd_layer[1].bias.data = bias.data.clone().to(device)

    return svd_layer.to(device)



def replace_module(model, name, new_module):
    """Replace a submodule given its dotted name, e.g. 'transformer.h.0.mlp.c_fc'."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)
    
    
    


import numpy as np


def compute_svd(weight_matrix):
    
    U,S,Vt = np.linalg.svd(weight_matrix,full_matrices=False)
    
    
    return U,S,Vt


def truncate_svd(U,S,Vt,rank):
    
    return U[:,:rank],S[:rank],Vt[:rank,:]


def reconstruct_matrix(U_k, S_k,Vt_k):
    
    return  (U_k * S_k) @ Vt_k


def compute_reconstruction_error(weight_matrix , W_compressed):
    
    return np.linalg.norm(weight_matrix-W_compressed)
    
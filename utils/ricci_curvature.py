import math
from numba import cuda
import numpy as np
import torch
from torch_geometric.utils import (
    to_dense_adj,
    remove_self_loops,
)

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:])"
)
def _balance_forman_curvature(A, A2, d_in, d_out, N, C):
    """Function for calculating the Ricci Curvature Matrix C.
    Args:
        A: Adjacency matrix of graph of interest 
        A2: A2 = A @ A which is the 2nd power of the adjacency matrix
        d_in: in_degree tensor of the graph
        d_out: out_degree tensor of the graph
        N: Number of nodes of the graph
        C: Ricci Curvature matrix; C_ij = Ric(i, j) for node i and node j
    """
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0: # if edge i->j does not exist
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[i]
            d_min = d_in[i]
        
        if d_max * d_min == 0:
            C[i, j] = 0
            return 

        sharp_ij = 0
        lambda_ij = 0
        for k in range(N):
            TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP
            TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP
        C[i, j] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j] 
        )
        if lambda_ij > 0:
            C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros(N, N).cuda()
    
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _balance_forman_curvature[blockspergrid, threadsperblock](A, A2, d_in, d_out, N, C)
    return C


def get_ricci_curvature(data):
    edge_index = data.edge_index
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    A = A.cuda()
    C = torch.zeros(N, N).cuda()
    balanced_forman_curvature(A, C=C)
    return C
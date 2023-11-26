import torch
import numpy as np
import torch.nn.init as init

def initialize_matrix(n_rows, n_cols):
    matrix = torch.empty(n_rows, n_cols)
    init.xavier_normal_(matrix)
    matrix.requires_grad = True
    return matrix

def stack_heads(n_heads, n_rows, n_cols):
    tensors = [initialize_matrix(n_rows, n_cols) for _ in range(n_heads)]
    return tensors

def self_attention(mask, Q, K, V):
    att = (Q @ K.T)/np.sqrt(Q.shape[0]) + mask
    return (torch.softmax(att, dim = 1) @ V)

def manually_update(tensor_list, lr):
    with torch.no_grad():
        for tensor in tensor_list:
            tensor -= lr * tensor.grad
    for tensor in tensor_list:
        tensor.grad.zero_()

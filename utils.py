import torch
import numpy as np

def z_score(matrix):
    mean = torch.mean(matrix, dim = 1, keepdim = True)
    std = torch.std(matrix, dim = 1, keepdim = True)
    z_score_vector = (matrix-mean)/std
    return z_score_vector

def self_attention(mask, Q, K, V):
    att = (Q @ K.T + mask)/np.sqrt(Q.shape[1])
    return (torch.softmax(att, dim = 0) @ V)
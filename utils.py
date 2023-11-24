import torch
import numpy as np

def z_score(matrix):
    mean = torch.mean(matrix, dim = 1, keepdim = True)
    std = torch.std(matrix, dim = 1, keepdim = True)
    epsilon = 1e-7
    z_score_vector = (matrix-mean)/(std+epsilon)
    return z_score_vector

def self_attention(mask, Q, K, V):
    att = (Q @ K.T + mask)/np.sqrt(Q.shape[0])
    return (torch.softmax(att, dim = 1) @ V)
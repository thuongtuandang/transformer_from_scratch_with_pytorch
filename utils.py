import torch
import numpy as np

def self_attention(Q, K, V, mask):
    att = (Q @ K.T)/np.sqrt(Q.shape[0]) + mask
    return (torch.softmax(att, dim = 1) @ V)
import torch
import numpy as np

def self_attention(mask, Q, K, V):
    att = (Q @ K.T)/np.sqrt(Q.shape[0]) + mask
    return (torch.softmax(att, dim = 1) @ V)
import torch
import numpy as np

def self_attention(mask, Q, K, V):
    att = (Q @ K.T + mask)/np.sqrt(Q.shape[0])
    return (torch.softmax(att, dim = 1) @ V)
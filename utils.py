import torch
import numpy as np

def self_attention(Q, K, V):
    att = (Q @ K.T)/np.sqrt(Q.shape[0])
    return (torch.softmax(att, dim = 1) @ V)
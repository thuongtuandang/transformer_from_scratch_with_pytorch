import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from dataset import Dataset
from utils import self_attention

class Transformer():
    # T = max_sentence_length
    # dim_input_size = dim of word embedding, in this case it is vocab_size
    def __init__(self, input_size, dim_input_size, output_size, hidden_size = 64):
        # That is the number of heads
        self.n_heads = 1
        # Here we can define matrices in pytorch with require_grad = True
        # Initialize matrices for self-attention layer
        self.Wq = torch.empty(dim_input_size, 0)
        self.Wk = torch.empty(dim_input_size, 0)
        self.Wv = torch.empty(dim_input_size, 0)
        for i in range(self.n_heads):
            Wq = torch.rand(dim_input_size, hidden_size)/1000
            self.Wq = torch.cat((self.Wq, Wq), dim = 1)
            Wk = torch.rand(dim_input_size, hidden_size)/1000
            self.Wk = torch.cat((self.Wk, Wk), dim = 1)
            Wv = torch.rand(dim_input_size, hidden_size)/1000
            self.Wv = torch.cat((self.Wv, Wv), dim = 1)
        
        self.Wq.requires_grad = True
        self.Wk.requires_grad = True
        self.Wv.requires_grad = True

        # Initialize the matrix Wo for the projection of the multi head layer output
        self.Wo = torch.rand(self.n_heads * hidden_size, hidden_size)/1000
        self.Wo.requires_grad = True
        # Create the mask matrix with upper part is -infinity
        # and other entries are 0
        # First we create a matrix filled with - infinity at all entries
        matrix = torch.full((input_size, input_size), -float('inf'))
        # triu = triangular upper part, we want to fill in upper part with - infinity
        matrix = torch.triu(matrix)
        # Replace the diagonal by 0
        matrix = matrix.fill_diagonal_(0)
        self.mask = matrix
        # That is for the positional encoding
        self.n = 10000

        # Initialize parameters for the feed forward layer
        self.Wf = torch.rand(output_size, input_size)/1000
        self.Wf.requires_grad = True
        self.bf = torch.zeros((output_size, 1), requires_grad = True)

        # Initialize for the linear layer
        self.Wl = torch.rand(hidden_size, 1)/1000
        self.Wl.requires_grad = True
        self.bl = torch.zeros((output_size, 1), requires_grad = True)    
    
    # Positional encoding for the whole sentence
    def positional_encoding(self, inputs):
        pos_encoding = torch.empty(0, inputs.shape[1])
        iter = inputs.shape[0]
        for k in range(iter):
            d = inputs.shape[1]
            pos = torch.zeros((1,d))
            for i in range(d):
                if i % 2 == 0:
                    pos[0][i] = np.sin(np.radians(k/(self.n**(i/d))))
                if i % 2 == 1:
                    pos[0][i] = np.cos(np.radians(k/(self.n**((i-1)/d))))
            v_pos = inputs[k, :] + pos
            pos_encoding = torch.cat((pos_encoding, v_pos), dim = 0)
        return pos_encoding/2
    
    # Multi-head attention
    def multi_head_attention(self, inputs):
        q = inputs @ self.Wq
        k = inputs @ self.Wk
        v = inputs @ self.Wv
        attention = self_attention(q, k, v)
        output = attention @ self.Wo
        return output

    # Feed forward layer
    def feed_forward(self, inputs):
        output = self.Wf @ inputs + self.bf
        output = torch.relu(output)
        return output
    
    def linear(self, inputs):
        return inputs @ self.Wl + self.bl

    def forward(self, inputs):
        # Inputs is a whole sentence at a time
        # And this sentence is represented by a matrix
        # The matrix consists of row vectors, each row is a word embedding
        np_inputs = np.array(inputs)
        torch_inputs = torch.tensor(np_inputs)
        torch_inputs = torch_inputs.to(torch.float32)
        pos_inputs = self.positional_encoding(torch_inputs)
        output_self_attention = self.multi_head_attention(pos_inputs)
        output_feed_forward = self.feed_forward(output_self_attention)
        output_linear = self.linear(output_feed_forward)
        y_pred = torch.softmax(output_linear, dim = 0)
        return y_pred
    
    def process(self, X, y, run_backward = False):
        accuracy = 0
        for x, y_true in zip(X,y):
            # x is a matrix of a sentence
            probs = self.forward(x)
            # True label
            true_index = int(y_true) 
            # Accuracy
            accuracy += int(torch.argmax(probs) == true_index) 
            if run_backward:
                y_true_torch = torch.zeros((2,1))
                y_true_torch[true_index] = 1
                self.optimizer.zero_grad()
                L = self.loss(probs, y_true_torch)
                L.backward()
                self.optimizer.step()
         # Accuracy 
        return float(accuracy/len(X))
    
    def fit(self, X, y, max_iter = 201, learning_rate = 0.001, print_period = 20):
        self.loss = nn.BCELoss()
        self.optimizer = opt.SGD([self.Wq, self.Wk, self.Wv, self.Wo, 
                                  self.Wf, self.bf, 
                                  self.Wl, self.bl],
                                  lr = learning_rate)
        for i in range(max_iter):
            accuracy = self.process(X, y, run_backward=True)
            if(i % print_period == 0):
                print(f"Step: {i}")
                print(f"accuracy for training data: {accuracy}")
    
    def predict(self, X, y):
        return self.process(X, y, run_backward=False)
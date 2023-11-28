import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from dataset import Dataset
from utils import initialize_matrix, stack_heads, self_attention, manually_update
import torch.nn.init as init
import math

class Transformer():
    # T = max_sentence_length
    # dim_input_size = dim of word embedding, in this case it is vocab_size
    def __init__(self, input_size, dim_input_size, output_size, hidden_size = 64):
        # That is the number of heads
        self.n_heads = 2
        # Here we can define matrices in pytorch with require_grad = True
        # Initialize matrices for self-attention layer
        self.Wq = stack_heads(self.n_heads, dim_input_size, hidden_size) 
        self.Wk = stack_heads(self.n_heads, dim_input_size, hidden_size)
        self.Wv = stack_heads(self.n_heads, dim_input_size, hidden_size)

        # Initialize the matrix Wo for the projection of the multi head layer output
        self.Wo = initialize_matrix(self.n_heads * hidden_size, hidden_size)

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

        # Gamma and beta is for the normalization layer
        self.gamma_self = initialize_matrix(dim_input_size + hidden_size, hidden_size)
        self.beta_self = torch.zeros((input_size,1), requires_grad=True)

        # Initialize parameters for the feed forward layer
        self.Wf = initialize_matrix(output_size, input_size)
        self.bf = torch.zeros((output_size, 1), requires_grad = True)

        # Initialize gamma, beta for layer normalization after feedforward layer
        self.gamma_feed = initialize_matrix(hidden_size, output_size)
        self.beta_feed = torch.zeros((input_size + output_size,1), requires_grad=True)

        # Initialize for the linear layer
        self.Wl = initialize_matrix(1, input_size + output_size)
        self.bl = torch.zeros((1,1), requires_grad = True)    
    
    # Positional encoding for the whole sentence
    # def positional_encoding(self, inputs):
    #     pos_encoding = torch.empty(0, inputs.shape[1])
    #     iter = inputs.shape[0]
    #     d = inputs.shape[1]
    #     position = torch.arange(iter).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    #     pe = torch.zeros(iter, d)
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     return pe + inputs

    # Positional encoding for the whole sentence
    def positional_encoding(self, inputs):
        pos_encoding = torch.empty(0, inputs.shape[1])
        iter = inputs.shape[0]
        for k in range(iter):
            d = inputs.shape[1]
            pos = torch.zeros((1,d))
            for i in range(d):
                if i % 2 == 0:
                    pos[0][i] = np.sin((k/(self.n**(i/d))))
                if i % 2 == 1:
                    pos[0][i] = np.cos((k/(self.n**((i-1)/d))))
            v_pos = inputs[k, :] + pos
            pos_encoding = torch.cat((pos_encoding, v_pos), dim = 0)
        return pos_encoding
    
    # Multi-head attention
    def multi_head_attention(self, inputs):
        # h is the cat matrix after we compute self attention of each head
        h = torch.empty(inputs.shape[0], 0)
        for i in range(len(self.Wq)):
            qi = inputs @ self.Wq[i]
            ki = inputs @ self.Wk[i]
            vi = inputs @ self.Wv[i]
            hi = self_attention(self.mask, qi, ki, vi)
            # We concatenate hi to h by stacking rows 
            h = torch.cat((h, hi), dim = 1)
        output = h @ self.Wo
        return output
        
    # Layer norm after self-attention layer
    def layer_norm_self_attention(self, previous_layer_outputs, previous_layer_inputs):
        cat_matrix = torch.cat((previous_layer_outputs, previous_layer_inputs), dim = 1)
        next_inputs = torch.softmax(cat_matrix @ self.gamma_self + self.beta_self, dim = 1)
        return next_inputs

    # Feed forward after layer norm
    def feed_forward(self, inputs):
        output = self.Wf @ inputs + self.bf
        output = torch.relu(output)
        return output
        
    # Layer norm after feed_forward
    def layer_norm_feed_forward(self, previous_layer_outputs, previous_layer_inputs):
        cat_matrix = torch.cat((previous_layer_outputs, previous_layer_inputs), dim = 0)
        next_inputs =  torch.softmax(cat_matrix @ self.gamma_feed + self.beta_feed, dim = 1)
        return next_inputs
    
    def linear(self, inputs):
        return self.Wl @ inputs + self.bl

    def forward(self, inputs):
        # Inputs is a whole sentence at a time
        # And this sentence is represented by a matrix
        # The matrix consists of row vectors, each row is a word embedding
        np_inputs = np.array(inputs)
        torch_inputs = torch.tensor(np_inputs)
        torch_inputs = torch_inputs.to(torch.float32)
        pos_inputs = self.positional_encoding(torch_inputs)
        output_self_attention = self.multi_head_attention(pos_inputs)
        output_layer_norm_self = self.layer_norm_self_attention(output_self_attention, pos_inputs)
        output_feed_forward = self.feed_forward(output_layer_norm_self)
        output_layer_norm_feed = self.layer_norm_feed_forward(output_feed_forward, output_layer_norm_self)
        output_linear = self.linear(output_layer_norm_feed)
        y_pred = torch.softmax(output_linear, dim = 1)
        return y_pred
    
    def process(self, X, y, run_backward = False, lr = None):
        accuracy = 0
        for x, y_true in zip(X,y):
            # x is a matrix of a sentence
            probs = self.forward(x)
            # True label
            true_index = int(y_true) 
            # Accuracy
            accuracy += int(torch.argmax(probs) == true_index) 
            if run_backward:
                y_true_torch = torch.zeros((1,2))
                y_true_torch[0][true_index] = 1
                L = self.loss(probs, y_true_torch)
                self.optimizer.zero_grad()
                L.backward()
                self.optimizer.step()
                # Update Wq, Wk, Wv manually
                manually_update(self.Wq, lr)
                manually_update(self.Wk, lr)
                manually_update(self.Wv, lr)
         # Accuracy 
        return float(accuracy/len(X))
    
    def fit(self, X, y, max_iter = 201, learning_rate = 0.03, print_period = 20):
        self.loss = nn.BCELoss()
        # Because Wq, Wk, Wv are tensor list
        # and we are not using nn.Module
        # We need to update those manually
        self.optimizer = opt.SGD([self.Wo, 
                                  self.gamma_self, self.beta_self,
                                  self.Wf, self.bf, 
                                  self.gamma_feed, self.beta_feed,
                                  self.Wl, self.bl],
                                  lr = learning_rate)
        for i in range(max_iter):
            accuracy = self.process(X, y, run_backward=True, lr=learning_rate)
            if(i % print_period == 0):
                print(f"Step: {i}")
                print(f"accuracy for training data: {accuracy}")
    
    def predict(self, X, y):
        return self.process(X, y, run_backward=False)
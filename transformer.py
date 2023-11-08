import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from dataset import Dataset
from utils import z_score, self_attention

class Transformer():
    def __init__(self, input_size, output_size, hidden_size = 64):
        # That is the number of heads
        self.n_heads = 2
        # Here we can define matrices in pytorch with require_grad = True
        # Initialize matrices for self-attention layer
        self.Wq = torch.empty(0, input_size)
        self.Wk = torch.empty(0, input_size)
        self.Wv = torch.empty(0, input_size)
        for i in range(self.n_heads):
            Wq = torch.rand(hidden_size, input_size)/1000
            self.Wq = torch.cat((self.Wq, Wq), dim = 0)
            Wk = torch.rand(hidden_size, input_size)/1000
            self.Wk = torch.cat((self.Wk, Wk), dim = 0)
            Wv = torch.rand(hidden_size, input_size)/1000
            self.Wv = torch.cat((self.Wv, Wv), dim = 0)
        
        self.Wq.requires_grad = True
        self.Wk.requires_grad = True
        self.Wv.requires_grad = True

        # Initialize the matrix Wo for the projection of the multi head layer output
        self.Wo = torch.rand(hidden_size, self.n_heads * hidden_size)/1000
        self.Wo.requires_grad = True
        # Create the mask matrix with upper part is -infinity
        # and other entries are 0
        # First we create a matrix filled with - infinity at all entries
        matrix = torch.full((self.n_heads*hidden_size, self.n_heads * hidden_size), -float('inf'))
        # triu = triangular upper part, we want to fill in upper part with - infinity
        matrix = torch.triu(matrix)
        # Replace the diagonal by 0
        matrix = matrix.fill_diagonal_(0)
        self.mask = matrix
        # That is for the positional encoding
        self.n = 10000


        # Dimension matching layer for the layer norm
        # This layer norm is after self-attention layer
        self.Wnorm_self = torch.rand(hidden_size, input_size)/1000
        self.Wnorm_self.requires_grad = True
        self.bnorm_self = torch.zeros((hidden_size,1), requires_grad=True)
        # Rows of gamma_self is equal to sentence_max_length
        ds = Dataset()
        sentence_max_length = ds.sentence_max_length
        self.gamma_self = torch.rand((sentence_max_length,hidden_size))/1000
        self.gamma_self.requires_grad = True
        self.beta_self = torch.zeros((hidden_size,1), requires_grad=True)

        # Initialize matrices for feed-forward layer
        self.Wf = torch.rand(output_size, hidden_size)/1000
        self.Wf.requires_grad = True
        self.bf = torch.zeros((output_size,1), requires_grad=True)

        # Initialize parameters for layer normalization after feedforward layer
        self.Wnorm_feed = torch.rand(output_size, hidden_size)/1000
        self.Wnorm_feed.requires_grad = True
        self.bnorm_feed = torch.rand((output_size,1), requires_grad=True)
        self.gamma_feed = torch.rand((hidden_size,1))/1000
        self.gamma_feed.requires_grad = True
        self.beta_feed = torch.zeros((output_size,1), requires_grad=True)    
    
    # Positional encoding for the whole sentence
    def positional_encoding(self, inputs):
        pos_encoding = torch.empty(inputs.shape[0],0)
        iter = inputs.shape[1]
        for k in range(iter):
            d = inputs.shape[0]
            pos = torch.zeros((d,1))
            for i in range(d):
                if i % 2 == 0:
                    pos[i] = np.sin(k/(self.n**(i/d)))
                if i % 2 == 1:
                    pos[i] = np.cos(k/(self.n**((i-1)/d)))
            v_pos = inputs[:, k].reshape(-1,1) + pos
            pos_encoding = torch.cat((pos_encoding, v_pos), dim = 1)
        return pos_encoding
    
    # Multi-head attention
    def multi_head_attention(self, inputs):
        q = self.Wq @ inputs
        k = self.Wk @ inputs
        v = self.Wv @ inputs
        attention = self_attention(self.mask, q, k, v)
        output = self.Wo @ attention
        return output
        
    # Layer norm after self-attention layer
    def layer_norm_self_attention(self, previous_layer_outputs, inputs):
        inputs_matching = self.Wnorm_self @ inputs + self.bnorm_self
        vector = previous_layer_outputs + inputs_matching
        z_score_vector = z_score(vector)
        next_inputs = z_score_vector @ self.gamma_self + self.beta_self
        return next_inputs

    # Feed forward after layer norm
    def feed_forward(self, inputs):
        output = self.Wf @ inputs + self.bf
        output = torch.relu(output)
        return output
        
    # Layer norm after feed_forward
    def layer_norm_feed_forward(self, previous_layer_outputs, inputs):
        intputs_matching = self.Wnorm_feed @ inputs + self.bnorm_feed
        vector = previous_layer_outputs + intputs_matching
        z_score_vector = z_score(vector)
        next_inputs = z_score_vector @ self.gamma_feed + self.beta_feed
        return next_inputs
    
    def forward(self, inputs):
        # Inputs is a whole sentence at a time
        # Inputs will be a list with only one element
        # And this element is the matrix representation of the sentence
        # The matrix consists of row vectors, each row is a word embedding
        torch_inputs = torch.tensor(inputs)
        torch_inputs = torch_inputs.T
        torch_inputs = torch_inputs.to(torch.float32)
        pos_inputs = self.positional_encoding(torch_inputs)
        output_self_attention = self.multi_head_attention(pos_inputs)
        output_layer_norm_self = self.layer_norm_self_attention(output_self_attention, pos_inputs)
        output_feed_forward = self.feed_forward(output_layer_norm_self)
        output_layer_norm_feed = self.layer_norm_feed_forward(output_feed_forward, output_layer_norm_self)
        y_pred = torch.softmax(output_layer_norm_feed, dim = 0)
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
                L = self.loss(probs, y_true_torch)
                self.optimizer.zero_grad()
                L.backward()
                self.optimizer.step()
         # Accuracy 
        return float(accuracy/len(X))
    
    def fit(self, X, y, max_iter = 1001, learning_rate = 0.001):
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = opt.SGD([self.Wq, self.Wk, self.Wv, self.Wo, 
                                self.Wnorm_self, self.bnorm_self, self.gamma_self, self.beta_self, 
                                self.Wf, self.bf, 
                                self.Wnorm_feed, self.bnorm_feed, self.gamma_feed, self.beta_feed],
                                lr = learning_rate)
        for i in range(max_iter):
            accuracy = self.process(X, y, run_backward=True)
            if(i % 100 == 0):
                print(f"Step: {i}")
                print(f"accuracy for training data: {accuracy}")
    
    def predict(self, X, y):
        return self.process(X, y, run_backward=False)
import pandas as pd
import numpy as np
import string

class Dataset:
    def __init__(self, input_length = 2, output_length = 2) -> None:
        self.input_length = input_length
        self.output_length = output_length
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.word_dict = []
        self.sentence_max_length = 10

    def df_to_dict(self, data_path):
        df = pd.read_csv(data_path)
        dict = df.set_index('X').T.to_dict('list')
        dct = {}
        for k, v in dict.items():
            label = str(v).translate(str.maketrans('', '', string.punctuation))
            if label == 'True':
                b = True
            if label == 'False':
                b = False
            dct[k] = b
        return dct
    
    def create_word_dict(self, dict_data):
        vocab = []
        for k, v in dict_data.items():
            v = k.split()
            for word in v:
                if word not in vocab:
                    vocab.append(word)
        self.vocab_size = len(vocab)
        self.word_dict = vocab
        self.input_length = self.vocab_size
    
    def word_to_idx(self, word):
        return self.word_dict.index(word)
    
    def idx_to_word(self, idx):
        return self.word_dict[idx]
    
    def OneHotEncode(self, word):
        v = np.zeros(self.input_length)
        v[self.word_to_idx(word)] = 1
        return v
    
    def createInputs(self, inputs):
        v = []
        for word in inputs.split():
            v.append(self.OneHotEncode(word))
        for i in range(self.sentence_max_length - len(v)):
            v.append(np.zeros(self.input_length))
        return v
    
    def create_X_y(self, dict):
        X = []
        y = []
        for k, v in dict.items():
            X.append(self.createInputs(k))
            y.append(v)
        return X, y
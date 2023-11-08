import pickle
from dataset import Dataset
import torch
import argparse
from experiments import train_data_path

from transformer import Transformer

def main(args):
    with open('saved_model/model.pkl', mode='rb') as f:
        model: Transformer = pickle.load(f)
        text = args.text
        dataset = Dataset()
        train_dict = dataset.df_to_dict(train_data_path)
        # This will create a word_dict consists of all words in train data
        dataset.create_word_dict(train_dict)
        try:
            v = dataset.createInputs(text)
            y_pred = model.forward(v)
            index = torch.argmax(y_pred)
            if index == 0:
                print("Negative feeling")
            if index == 1:
                print("Positive feeling")
        except ValueError:
            print(f"The text '{text}' has words that are not in dictionary of the data file ({train_data_path}). Please try another words.")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Experiment model"
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)
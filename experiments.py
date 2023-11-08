from dataset import Dataset
from transformer import Transformer
import argparse
import pickle

train_data_path = "data/train_data.csv"
test_data_path = "data/test_data.csv"

def main(args):
    dataset = Dataset()
    # Train_dict is a dictionary of train_data
    train_dict = dataset.df_to_dict(train_data_path)
    # This will create a word_dict consists of all words in train data
    dataset.create_word_dict(train_dict)
    # This will create a big matrix of all sentences
    # Each element in X_train is a matrix of a sentence
    # Each element in y_train is a boolean value
    X_train, y_train = dataset.create_X_y(train_dict)

    model = Transformer(dataset.vocab_size, dataset.output_length, args.hidden_size)
    model.fit(X_train, y_train, max_iter = args.num_iter, learning_rate = args.learning_rate)

    test_dict = dataset.df_to_dict(test_data_path)
    X_test, y_test = dataset.create_X_y(test_dict)
    acc = model.predict(X_test, y_test)
    print(f"Accuracy for test set: {acc}")
    with open('saved_model/model.pkl', mode='wb') as file:
        pickle.dump(model, file)
    print("Model is saved at saved_model/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment model"
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        type = int,
        required = False,
        default = 64
    )
    parser.add_argument(
        "-ni",
        "--num_iter",
        type = int,
        required = False,
        default = 201
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type = float,
        required = False,
        default = 0.001
    )
    args = parser.parse_args()
    main(args)
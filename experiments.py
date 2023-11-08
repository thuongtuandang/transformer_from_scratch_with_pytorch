from dataset import Dataset
from transformer import Transformer

train_data_path = "data/train_data.csv"
test_data_path = "data/test_data.csv"

def main():
    dataset = Dataset()
    # Train_dict is a dictionary of train_data
    train_dict = dataset.df_to_dict(train_data_path)
    # This will create a word_dict consists of all words in train data
    dataset.create_word_dict(train_dict)
    # This will create a big matrix of all sentences
    # Each element in X_train is a matrix of a sentence
    # Each element in y_train is a boolean value
    X_train, y_train = dataset.create_X_y(train_dict)

    model = Transformer(dataset.vocab_size, dataset.output_length)
    model.fit(X_train, y_train)

    test_dict = dataset.df_to_dict(test_data_path)
    X_test, y_test = dataset.create_X_y(test_dict)
    model.predict(X_test, y_test)

if __name__ == "__main__":
    main()




# transformer_from_scratch_with_pytorch
## Goal of this project
- Implement a transformer model from scratch with Pytorch.
- Provide a complete documentation about the theoritical aspetcs of transformer mechanism with sample codes.
- Process text data and transform it into a form useful for our model for the prediction task.
- Predict sentiment based on text data.

## Documentation
We highly recommend you to take a look on our documentation file `Documentation/Transformer.pdf`.
In this documentation, we carefully treated every theoritical aspect in the paper 'Attention is all you need' in detail with codes.

Interesting points contained in the file:
- The difference of the positional encoding between time t and time (t + k) is a rotation of complex numbers.
- Self-attention really tells us how the current input depends on other inputs.
- Residual connections can reduce random noises and the possibility of vanishing gradient.
- Step by step implement them in Python.

## Install python package
### Using `pipenv`:
```
pipenv install
```
## Training
You can train a transformer model with the default setting:
- train data: `data/train_data.csv`
- test data: `data/test_data.csv`
- learning rate: `0.03`
- number of iterations: `201`
- print period: `20`
- hidden size: `18`
```
python experiments.py
```
You can switch to your own data by coping your data file to the `data` folder and update the `train_data_path` and `test_data_path` variables in the `experiments.py` file.
```
train_data_path = "path/to/your/train/data/file"
test_data_path = "path/to/your/test/data/file"
```
You can also change the `learning rate`, the `number of iterations` and the `print period` with this command:
```
python experiments.py --num_iter 201 --learning_rate 0.03 --print_period 20
```
## Prediction
After training, the trained model is saved at `saved_model/model.pkl` so that we can run the prediction without retraining by simply loading the model file. You can run the prediction by the following command:
```
python sentiment_prediction.py --text "i am happy"
```
with the `--text` param is the sentence you want to predict sentiment.

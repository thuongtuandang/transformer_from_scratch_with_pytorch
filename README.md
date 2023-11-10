# transformer_from_scratch_with_pytorch
## Goal of this project
- Implement a transformer model from scratch with Pytorch.
- Provide a complete documentation about the theoritical aspetcs of transformer mechanism with sample codes.
- Process text data and transform it into a form useful for our model for the prediction task.
- Predict sentiment based on text data.

## Documentation
We highly recommend you to take a look on our documentation file `documents/Transformer.pdf`.
In this documentation, we carefully treated every theoritical aspect in the paper 'Attention is all you need' in detail.

Interesting points contained in the file:
- The difference of the positional encoding between time t and time (t + k) is a rotation of complex numbers.
- Self-attention really tells us how the current input depends on other inputs.
- Residual connections can reduce random noises and the possibility of vanishing gradient.

## Install python package
### Using `venv`:
```
python -m venv ./myenv
source ./myenv/bin/activate
pip install requirements.txt
```
### Using `pipenv`:
```
pipenv install
```
## Training
You can train a transformer model with the default setting:
- train data: `data/train_data.csv`
- test data: `data/test_data.csv`
- learning rate: `0.001`
- number of iterations: `201`
```
python experiments.py
```
You can switch to your own data by coping your data file to the `data` folder and update the `DATA_PATH` variable in the `experiments.py` file.
```
DATA_PATH = "path/to/your/data/file"
```
You also can change the `learning rate` and the `number of iterations` with this command:
```
python experiments.py --learning_rate 0.001 --num_iter 201
```
## Prediction
After training, the trained model is saved at `saved_model/model.pkl` so that we can run the prediction without retraining by simply loading the model file. You can run the prediction by the following command:
```
python word_prediction.py --text "i am happy"
```
with the `--text` param is the sentence you want to predict sentiment.

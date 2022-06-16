import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MKLDNN_VERBOSE"] = "1"

model = load_model('./asa_model.h5')


def predict_from_word(word):

    with open('asa_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Create a token for the work
    token = tokenizer.texts_to_sequences([word])
    tw = pad_sequences(token, maxlen=256)

    # Predict with the model
    prediction = model.predict(tw).item()
    print('\n', word, '\n', prediction)

    return prediction


@app.route('/', methods=['GET'])
def home():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    word = request.json['word']
    prediction = 0
    try:
        prediction = predict_from_word(word)
        return jsonify({'word': word, 'prediction': prediction})
    except Exception as e:
        return jsonify({'Error': e})


if __name__ == '__main__':
    app.run(debug=True)

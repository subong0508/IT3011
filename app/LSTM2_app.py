import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import sys, os

sys.path.append(os.pardir)

import pickle
import getXY_new

with open('getXY_new.pickle', 'rb') as f:
  get = pickle.load(f)

tokenizer = get.get_tokenizer()
lstm_clf = load_model('LSTM_model2.h5')

label_emoji_mapping, emoji_label_mapping = get.get_mappings()

app = Flask(__name__)

class ReviewForm(Form):
    text = TextAreaField('', [validators.DataRequired(), validators.length(min=10)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('form.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['text']
        print(review)
        sequences = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequences, padding = 'post', maxlen = 26)
        labels = np.argsort(-lstm_clf.predict(padded))[:, :3].reshape(-1, 1)
        emojis = []

        for x in labels[:,0]:
            emojis.append(label_emoji_mapping[x])

        emojis = ','.join(emojis)

        return render_template('results.html',
                                content=review,
                                prediction= emojis)
    return render_template('form.html', form=form)

if __name__ == "__main__":
  app.run()
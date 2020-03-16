import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import warnings
import pickle

warnings.filterwarnings('ignore')

all_tweets = pd.read_csv("data/emojis.csv")
unique_emoji = all_tweets.emoji.unique()

emoji_counts = all_tweets['emoji'].value_counts().head(20)
emoji_counts = emoji_counts.to_frame()
emoji_list = np.array(emoji_counts.index)

df = all_tweets.loc[all_tweets['emoji'].isin(emoji_list)]

pattern = r'(http://[^"\s]+)|(@\w+)|(:)|([^\w\d\s\.\?\!])'

df.loc[:, "text"] = df.text.str.replace(pattern, "").values

train_sentences, test_sentences, train_emojis, test_emojis = train_test_split(df['text'], df['emoji'], stratify = df['emoji'],
                                                                             random_state = 3011) # for consistency
max_string = max(df['text'], key=lambda t:len(t))
max_string_len = len(re.split(r'[\s+^\.]', max_string))

actual_emoji = df['emoji'].value_counts().index.values
label_emoji_mapping = dict([(label, emoji) for label, emoji in zip(range(20), actual_emoji)])
emoji_label_mapping = dict([(emoji, label) for label, emoji in label_emoji_mapping.items()])

y_train, y_test = train_emojis.replace(emoji_label_mapping), test_emojis.replace(emoji_label_mapping)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

class getXY_new():
    def __init__(self):
        pass
    
    def get_train_set(self):
        return train_padded, y_train_onehot
    
    def get_test_set(self):
        return test_padded, y_test_onehot

with open('getXY_new.pickle', 'wb') as f:
    pickle.dump(getXY_new(), f)

VOCAB_SIZE = 10000
tokenizer = Tokenizer(oov_token = 'OOV', num_words = VOCAB_SIZE) # revised
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding = 'post', maxlen = max_string_len)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding = 'post', maxlen = max_string_len)
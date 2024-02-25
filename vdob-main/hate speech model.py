#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[2]:


#Basic
import numpy as np
import pandas as pd
import logging

#Data Preprocessing
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Saving
import joblib
import shutil

#Models
from sklearn.ensemble import RandomForestClassifier
import catboost
from catboost import CatBoost
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

#Neural Network
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#TuningS
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer_names
from sklearn.metrics import make_scorer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import load_model

#Feature Selection
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

#Settings
pd.set_option('display.max_rows',None)


# In[4]:


## Upload Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X = train['comment_text']
y = train.copy()
drop_columns = ['id','comment_text']
y = (y.drop(columns=drop_columns, axis=0)).astype(int)

Xtest = test['comment_text']


# In[5]:


train.head()


# In[6]:


train['comment_text'][0]


# In[ ]:


max_words = 25000
max_len = 500

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen = max_len)

Xtest_seq = tokenizer.texts_to_sequences(Xtest)
Xtest_pad = pad_sequences(Xtest_seq, maxlen = max_len)


# In[ ]:


## Text CNN Architecture

embedding_dim = 60

model = models.Sequential()

# Embedding layer
model.add(layers.Embedding(max_words, embedding_dim, input_length=max_len)) #vocab size, embed vector size, input size

# Convolutional layers with different filter sizes
model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))

# GlobalMaxPooling
model.add(layers.GlobalMaxPooling1D())

# Dense Layer
model.add(layers.Dense(units=20, activation='relu'))

# Output layer for classification
model.add(layers.Dense(units=6, activation='sigmoid'))

# Define the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


## Train Model

X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size = 0.2, random_state=0)

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=[X_val, y_val])


# In[ ]:


## Evaluate Model

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')b


# In[ ]:


# Predictions
predictions = model.predict(Xtest_pad)
print(predictions)

predictions = pd.DataFrame(predictions)
output = pd.DataFrame(data={'id': test['id'],'toxic': predictions[0], 'severe_toxic': predictions[1],
                           'obscene': predictions[2], 'threat': predictions[3], 'insult': predictions[4],
                           'identity_hate': predictions[5]})
output.to_csv('data/submission.csv', index=False)


# In[ ]:


## CV Modelling Scores

def create_model():
    model = models.Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=max_len)) #vocab size, embed vector size, input size
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(units=20, activation='relu'))
    model.add(layers.Dense(units=6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = create_model()
        self.model.fit(X, y, **kwargs)

    def predict(self, X, threshold=0.5):
        y_pred_continuous = self.model.predict(X)
        y_pred_binary = (y_pred_continuous > threshold).astype(int)
        return y_pred_binary
    
    def predict_proba(self, X):
        y_pred_continuous = self.model.predict(X)
        probabilities = y_pred_continuous
        return probabilities

def hamming_score(y_true, y_pred):
    return 1 - hamming_loss(y_true, y_pred)

scoring = {
    'hamming_score': make_scorer(hamming_score),
    'roc_auc_ovr': 'roc_auc_ovr',
    'f1_weighted': 'f1_weighted'
}

keras_wrapper = KerasWrapper()

scores = cross_validate(keras_wrapper, X_pad, y, cv=5,
                       scoring=scoring,
                       return_train_score=True, error_score='raise')

avg_train_hamming_score = np.mean(scores['train_hamming_score'])
avg_train_roc_auc_ovr = np.mean(scores['train_roc_auc_ovr'])
avg_train_f1_weighted = np.mean(scores['train_f1_weighted'])
avg_test_hamming_score = np.mean(scores['test_hamming_score'])
avg_test_roc_auc_ovr = np.mean(scores['test_roc_auc_ovr'])
avg_test_f1_weighted = np.mean(scores['test_f1_weighted'])
avg_fit_time = np.mean(scores['fit_time'])
avg_score_time = np.mean(scores['score_time'])

# print('No. of Features Kept:', num_features_to_keep)
print("Average Train Hamming Score:", format(avg_train_hamming_score, '.5f'))
print("Average Train ROC AUC OVR Score:", format(avg_train_roc_auc_ovr, '.5f'))
print("Average Train F1 Weighted Score:", format(avg_train_f1_weighted, '.5f'))
print("Average Test Hamming Score:", format(avg_test_hamming_score, '.5f'))
print("Average Test ROC AUC OVR Score:", format(avg_test_roc_auc_ovr, '.5f'))
print("Average Test F1 Weighted Score:", format(avg_test_f1_weighted, '.5f'))
print("Average Fit Time (seconds):", format(avg_fit_time, '.5f'))
print("Average Score Time (seconds):", format(avg_score_time, '.5f'))


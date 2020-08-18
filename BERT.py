#!/usr/bin/env python

import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split


# Used for Loading Files later

training_file = f'train.csv'
testing_file = f'test.csv'


# Loading Data

train = pd.read_csv(training_file)
test = pd.read_csv(testing_file)


# Preprocessing the Data

# Remove Punctuation

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

train['comment_text'] = train['comment_text'].apply(remove_punctuation)
test['comment_text'] = test['comment_text'].apply(remove_punctuation)


# Remove Stop Words

sw=stopwords.words('english')

def removesw(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

train['comment_text'] = train['comment_text'].apply(removesw)
test['comment_text'] = test['comment_text'].apply(removesw)


# Applying Stemming

stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 
train['comment_text'] = train['comment_text'].apply(stemming)
test['comment_text'] = test['comment_text'].apply(stemming)


# Create DistilBERT model for Multi-label Classification

model = MultiLabelClassificationModel(
    "distilbert",
    "distilbert-base-uncased",
    num_labels=6,
    use_cuda = False,
    args={"train_batch_size": 4, "gradient_accumulation_steps": 16, "learning_rate":3e-5, "num_train_epochs": 2, "max_seq_length": 150, "reprocess_input_data": True, "overwrite_output_dir": True},
)

# Create compatible dataframe

train['labels'] = list(zip(train.toxic.tolist(), train.severe_toxic.tolist(), train.obscene.tolist(), train.threat.tolist(),  train.insult.tolist(), train.identity_hate.tolist()))
train['text'] = train['comment_text'].apply(lambda x: x.replace('\n', ' '))

train.head()


# Split dataset for validation
train_df, eval_df = train_test_split(train, test_size = 0.2)

model.train_model(train_df, output_dir="outputs/")


# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Create Submission for Testing on Kaggle

test_df = pd.read_csv(testing_file)

to_predict = test_df.comment_text.apply(lambda x: x.replace('\n', ' ')).tolist()
preds, outputs = model.predict(to_predict)

# Create submission data frame
submission_df = pd.DataFrame(model_outputs, columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
submission_df['id'] = test_df['id']
submission_df = submission_df[['id', 'toxic','severe_toxic','obscene','threat','insult','identity_hate']]

# Convert dataframe into CSV for submitting to Kaggle
submission_df.to_csv('outputs/submission.csv', index=False)


# Credits:
# 
# We used the SimpleTransformers library to use pre-trained models in an easy way.
# https://github.com/ThilinaRajapakse/simpletransformers






import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GRU, Dense
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data.csv")
smiles = df.iloc[:,0]
targets = df.iloc[:,1:13]
smiles_train, smiles_test, targets_train, targets_test = train_test_split(smiles, targets, test_size=0.2)

batch_size = 32
tokenizer = Tokenizer(filters='', lower=False, char_level=True)
tokenizer.fit_on_texts(smiles.values)
one_hot_train = tokenizer.texts_to_sequences(smiles_train.values)
one_hot_test = tokenizer.texts_to_sequences(smiles_test.values)
for index, i in enumerate(one_hot_train):
    one_hot_train[index] = np.pad(np.array(i), (0, 342 - len(one_hot_train[index])), mode='constant', constant_values=0)
one_hot_train = np.array(one_hot_train)
for index, i in enumerate(one_hot_test):
    one_hot_test[index] = np.pad(np.array(i), (0, 342 - len(one_hot_test[index])), mode='constant', constant_values=0)
one_hot_test = np.array(one_hot_test)

model = Sequential()
model.add(Embedding(len(tokenizer.index_docs) + 1, 50))
model.add(Conv1D(filters=192, kernel_size=3, strides=1))
model.add(GRU(units=224, return_sequences=True))
model.add(GRU(units=384))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(one_hot_train, targets_train, epochs=100, validation_split=0.2)

score = model.evaluate(one_hot_test, targets_test)
print(score)

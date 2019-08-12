
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GRU, Dense
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard


# In[3]:


df = pd.read_csv("./data.csv")


# In[18]:


smiles = df.iloc[:,0]
targets = df.iloc[:,1:13]
targets.fillna(2, inplace=True)
smiles_train, smiles_test, targets_train, targets_test = train_test_split(smiles, targets, test_size=0.2)


# In[26]:


batch_size = 32
tokenizer = Tokenizer(filters='', lower=False, char_level=True)
tokenizer.fit_on_texts(smiles.values)
one_hot_train = tokenizer.texts_to_sequences(smiles_train.values)
one_hot_test = tokenizer.texts_to_sequences(smiles_test.values)
one_hot_train = pad_sequences(one_hot_train, padding='post')
one_hot_test = pad_sequences(one_hot_test, padding='post')


# In[27]:


model = Sequential()
model.add(Embedding(len(tokenizer.index_docs) + 1, 50, input_length=one_hot_train.shape[1]))
model.add(Conv1D(filters=192, kernel_size=3, strides=1))
model.add(GRU(units=224, return_sequences=True))
model.add(GRU(units=384))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
tensorboardCallback = TensorBoard()


# In[29]:


model.fit(one_hot_train, targets_train, epochs=100, validation_split=0.2, callbacks=[tensorboardCallback])
score = model.evaluate(one_hot_test, targets_test)
print(score)
model.save('my_model.h5')


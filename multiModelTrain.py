#!/usr/bin/env python
# coding: utf-8

# In[21]:


import tensorflow as tf
import pandas as pd
import numpy as np
import scanpy as sc 
import anndata
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


model = tf.keras.models.load_model('./model.cnn/')


# In[3]:


adata = anndata.read_h5ad('/scratch/nandivada.s/adata.h5ad')


# In[4]:


y = adata.obs['leiden']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(adata, y, test_size=0.2, random_state=42)
X_train = X_train.to_df()
X_test = X_test.to_df()
num_features = X_train.shape[1]


# In[6]:


X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# In[7]:


y_train = pd.DataFrame(to_categorical(y_train))
y_test = pd.DataFrame(to_categorical(y_test))


# In[22]:


# np.save('./2700Cells/X_train.npy',X_train)
# np.save('./2700Cells/y_train.npy',y_train)
# np.save('./2700Cells/X_test.npy',X_test)
# np.save('./2700Cells/y_test.npy',y_test)


# In[9]:


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)


# In[10]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Limit TensorFlow to use the first GPU only
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


# In[18]:


print(model.layers[0].input_shape)


# In[27]:


# input_layer = model.layers[0]

# input_shape = input_layer.input_shape

# n = input_shape[1] - X_train.shape[1]
# adata_train_padded = np.pad(X_train, (0, n), mode='constant')

# # Print the padded array
# print("Original array:", X_train.shape)
# print("Padded array:", adata_train_padded.shape)


# In[26]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stopping])


# In[ ]:


model.save('old_and_new.cnn')


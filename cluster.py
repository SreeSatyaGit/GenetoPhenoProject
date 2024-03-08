#!/usr/bin/env python
# coding: utf-8

# In[52]:


import tensorflow as tf
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import scanpy as sc
import numpy as np
import pandas as pd
import sys
import os


# In[8]:


csv_data = pd.read_csv('/scratch/nandivada.s/GSE100866_PBMC_vs_flow_10X-ADT_clr-transformed.csv',header='infer',index_col=0)


# In[9]:


cdt = np.transpose(csv_data)


# In[33]:


# import gzip

# input_file = '/Users/bharadwajanandivada/Downloads/filtered_feature_bc_matrix 2/barcodes.tsv.gz'
# output_file = './barcodes/barcodes.csv'

# with gzip.open(input_file, 'rb') as f_in:
#     with open(output_file, 'wb') as f_out:
#         f_out.write(f_in.read())

# print("File extracted successfully.")


# In[3]:


# adata = sc.read_10x_mtx(
#     '/Users/bharadwajanandivada/Downloads/filtered_feature_bc_matrix 2/',  # the directory with the `.mtx` file
#     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
#     cache=True)

# adata


# In[56]:


# X_train = np.load('/scratch/nandivada.s/X_train.npy')
# X_test = np.load('/scratch/nandivada.s/X_test.npy')
# y_train = np.load('/scratch/nandivada.s/y_train.npy')
# y_test = np.load('/scratch/nandivada.s/y_test.npy')

# X_train = np.load('./X_train.npy')
# X_test = np.load('./X_test.npy')
# y_train = np.load('./y_train.npy')
# y_test = np.load('./y_test.npy')
y = ['CD3','CD4','CD8','CD2','CD45RA','CD57','CD16','CD14','CD11c','CD19']


# In[57]:


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the labels
label_encoder.fit(y)

# Transform the labels to numeric values
y_numeric = label_encoder.transform(y)

# Print the transformed labels
print(y_numeric)


# In[59]:


X_train, X_test = train_test_split(cdt, test_size=0.2, random_state=42)


# In[63]:


print(X_test.shape)


# In[67]:


y_train = np.random.randint(0, 10, size=X_train.shape[0])
print(y_train)

y_test = np.random.randint(0, 10, size=X_test.shape[0])
print(y_test)


# In[79]:


num_features = cdt.shape[1]
print(num_features)


# In[71]:


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)


# In[72]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Limit TensorFlow to use the first GPU only
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPU")
        print(tf.config.list_physical_devices('GPU'))
    except RuntimeError as e:
        print(e)


# In[ ]:


model = Sequential()
model.add(Conv1D(filters=32 ,kernel_size=1, activation='relu', input_shape=(num_features,1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax')) 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stopping])


# In[ ]:


model.save('Protien_model.cnn')


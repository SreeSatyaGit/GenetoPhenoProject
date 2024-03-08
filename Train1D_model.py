#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Conv1D,Conv3D,Conv2D,MaxPooling1D,GlobalMaxPooling1D,MaxPooling2D,MaxPooling3D,Flatten,Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os


mod_data = pd.read_csv('/Users/bharadwajanandivada/Downloads/GSE100866_PBMC_vs_flow_10X-RNA_umi_8k_points.csv',header='infer',index_col=0)
hann_data = pd.read_csv('/Users/bharadwajanandivada/Downloads/GSE100866_PBMC_vs_flow_10X-RNA_umi_hanning.csv',header='infer',index_col=0)

labels10 = pd.read_csv('/Users/bharadwajanandivada/Downloads/GSE100866_PBMC_vs_flow_10X-ADT_clr-transformed.csv',header='infer',index_col=0)
y10 = np.transpose(labels10)


X_train,X_test,y_train,y_test = train_test_split(mod_data,y10,test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)

new_model = Sequential()
new_model.add(Conv1D(filters=32 ,kernel_size=1, activation='relu', input_shape=(X_train.shape,1)))
new_model.add(MaxPooling1D(pool_size=1))
new_model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
new_model.add(MaxPooling1D(pool_size=1))
new_model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
new_model.add(Flatten())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dense(64, activation='relu'))
new_model.add(Dense(10,activation = 'linear'))

new_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
new_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stopping])
new_model.save('model_mod_8k.cnn')
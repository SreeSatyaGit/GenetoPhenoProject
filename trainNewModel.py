
#!/usr/bin/env python
#coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Conv3D,Conv2D,MaxPooling1D,GlobalMaxPooling1D,MaxPooling2D,MaxPooling3D,Flatten,Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os



data = pd.read_csv('./data/GSE100866_PBMC_vs_flow_10X-RNA_umi_8k_points_blackman.csv',header='infer',index_col=0)
# # Directory path containing the images
# directory = './whiteSpaceRm/'

# # List to store the loaded images
# image_data = []

# # Iterate over the files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Construct the file path
#         file_path = os.path.join(directory, filename)
#         # Open the image file using PIL
#         image = Image.open(file_path)
#         # Append the image to the list
#         image_arr = np.array(image)
#         image_arr = np.resize(image_arr,(32,32,3))
#         image_data.append(image_arr/255)




labels10 = pd.read_csv('./data/GSE100866_PBMC_vs_flow_10X-ADT_clr-transformed.csv',header='infer',index_col=0)
y10 = np.transpose(labels10)

X_train, X_test,y_train,y_test = train_test_split(data,y10,test_size=0.2, random_state=42)


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)

model = Sequential()
model.add(Conv1D(filters=32 ,kernel_size=1, activation='relu', input_shape=(X_train.shape[0])))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10,activation = 'linear'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

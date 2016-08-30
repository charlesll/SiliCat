# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:58:15 2016

@author: charles
"""
#%%
import os
import numpy as np
import pickle as pkl

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

from keras.layers.advanced_activations import PReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import np_utils

from keras.regularizers import l1, l1l2, activity_l1l2

from sklearn import cross_validation, preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt

#import silicat
import scipy

#%%
spectra_liste = np.genfromtxt("./data/raman/spectra_labels.csv",delimiter=',',skip_header=1,dtype = 'string')# reading the list
features = spectra_liste[:,2:12].astype(np.float)
features[:,0:9] = features[:,0:9]*np.reshape((100.-features[:,9]),(len(features),1))/100.0

f=open('./data/raman/spectra_2d_supervised.pkl','r')
spectra = pkl.load(f) # dimension 1 in spectra equal dimention 1 in feature
f.close()

#%%
# getting index for the frames with the help of scikitlearn
names_idx = np.arange(len(spectra))
frame1_idx, frame2_idx = cross_validation.train_test_split(names_idx, test_size = 0.2,random_state=42)

# and now grabbing the relevant pandas dataframes
X_train = spectra[frame1_idx,:,:]
X_validation = spectra[frame2_idx,:,:]

y_train = features[frame1_idx,:]/100.
y_validation = features[frame2_idx,:]/100.

#%% Here we go
batch_size = 10
nb_classes = 10
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 100, 1000
# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (6, 6)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)

#%%
#model = model_from_json(open(os.path.dirname(os.path.abspath(__file__))+'/pretrained_ConvNet.json').read())
#model.load_weights(os.path.dirname(os.path.abspath(__file__))+'/ConvNet_pretrained_weights.h5')

# straight model to find the shapes
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',input_shape=(1, img_rows, img_cols)))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1],border_mode='same',activation='relu'))
#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

shape_endconv = model.output_shape
model.add(Flatten())
shape_flat = model.output_shape
model.add(Dense(128, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(nb_classes, init='lecun_uniform'))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_validation, y_validation),
          callbacks=[early_stopping])
          
score = model.evaluate(X_validation, y_validation, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

test_validation = model.predict(X_validation)
test_train = model.predict(X_train)

mse_si_train = np.sqrt(1.0/np.size(y_train,0)*np.sum(((y_train[:,0]-test_train[:,0])**2)))*100
mse_si_valid = np.sqrt(1.0/np.size(y_validation,0)*np.sum(((y_validation[:,0]-test_validation[:,0])**2)))*100

print(mse_si_train)
print(mse_si_valid)
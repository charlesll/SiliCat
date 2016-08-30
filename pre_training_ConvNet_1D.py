# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:58:15 2016

@author: charles
"""
#%%
import numpy as np
import pickle as pkl

from keras.models import Sequential, Model, model_from_json
#from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D, Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l1, l1l2, activity_l1l2

from sklearn import cross_validation, preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt

import silicat
import scipy

#%%
f=open('./data/raman/spectra_1d_unsupervised','r')
spectra = pkl.load(f) # dimension 1 in spectra equal dimention 1 in feature
f.close()

#%%
# getting index for the frames with the help of scikitlearn
names_idx = np.arange(len(spectra))
frame1_idx, frame2_idx = cross_validation.train_test_split(names_idx, test_size = 0.2)

# and now grabbing the relevant pandas dataframes
X_train = spectra[frame1_idx,:]
X_validation = spectra[frame2_idx,:]

#%% Here we go
batch_size = 8
nb_classes = 10
nb_epoch = 12

# input image dimensions
signal_freq = 1000
# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
pool_len = 2
# convolution kernel size
kernel_length = 3

X_train = X_train.reshape(X_train.shape[0], signal_freq, 1)
X_validation = X_validation.reshape(X_validation.shape[0], signal_freq, 1)

#%%

# straight model to find the shapes
model = Sequential()
model.add(Convolution1D(nb_filters, kernel_length, border_mode='same',activation='relu',input_dim=signal_freq))
print(model.output_shape)
model.add(MaxPooling1D(pool_length=pool_len))
print(model.output_shape)

model.add(Convolution1D(nb_filters, kernel_length, border_mode='same',activation='relu'))
model.add(MaxPooling1D(pool_length=pool_len))
print(model.output_shape)
shape_endconv = model.output_shape

input_sp = Input(shape = (signal_freq,1))    

x = Convolution1D(nb_filters, kernel_length, border_mode='same',activation='relu')(input_sp)
x = MaxPooling1D(pool_length=pool_len)(x)

x = Convolution1D(nb_filters*2, kernel_length, border_mode='same',activation='relu')(x)
encoded = MaxPooling1D(pool_length=pool_len)(x)

x = Convolution1D(nb_filters*2, kernel_length, border_mode='same',activation='relu')(encoded)
x = UpSampling1D(length=pool_len)(x)

x = Convolution1D(nb_filters, kernel_length, border_mode='same',activation='relu')(x)
x = UpSampling1D(length=pool_len)(x)

decoded = Convolution1D(1, kernel_length, border_mode='same',activation='sigmoid')(x)

autoencoder = Model(input_sp, decoded)
encoder = Model(input_sp,encoded)

#%%
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(X_train, X_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_validation, X_validation))

#%%      
test = autoencoder.predict(X_validation)






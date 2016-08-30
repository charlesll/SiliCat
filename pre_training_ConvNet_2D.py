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

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Activation, Flatten, Reshape, Input

from keras.layers.advanced_activations import PReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import np_utils

from keras.regularizers import l1, l1l2, activity_l1l2

from sklearn import cross_validation

import matplotlib
from matplotlib import pyplot as plt

#%%
f=open('./data/raman/spectra_2d_unsupervised.pkl','r')
spectra = pkl.load(f) # dimension 1 in spectra equal dimention 1 in feature
f.close()

#%%
# getting index for the frames with the help of scikitlearn
names_idx = np.arange(len(spectra))
frame1_idx, frame2_idx = cross_validation.train_test_split(names_idx, test_size = 0.2)

# and now grabbing the relevant pandas dataframes
X_train = spectra[frame1_idx,:,:]
X_validation = spectra[frame2_idx,:,:]

#%% Here we go
batch_size = 10
nb_classes = 10
nb_epoch = 30

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

# Now we do the same with a detailled model

input_img = Input(shape=(1, img_rows, img_cols))

#ENCODER
x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='glorot_uniform')(input_img)
x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)

x = Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='glorot_uniform')(x)
x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)

#x = Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='he_normal')(x)
#x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)

x=Flatten()(x)

x = Dense(128, init='lecun_uniform',activation='relu')(x)

encoded = Dense(nb_classes, init='lecun_uniform',activation='linear')(x) # the symmetry axis

# DECODER
x = Dense(128, init='lecun_uniform',activation='relu')(encoded)

x = Dense(shape_flat[1], init='he_normal',activation='relu')(x)

x = Reshape(shape_endconv[1:4])(x)

#x = Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='he_normal')(x)
#x = UpSampling2D((2,2))(x)

x = Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='glorot_uniform')(x)
x = UpSampling2D((2,2))(x)

x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='same',activation='relu',init='glorot_uniform')(x)
x = UpSampling2D((2,2))(x)

decoded = Convolution2D(1, kernel_size[0], kernel_size[1],border_mode='same',activation='linear',init='lecun_uniform')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# save as JSON for saving the architecture
json_string = encoder.to_json()
open('pretrained_ConvNet.json', 'w+').write(json_string)

json_string = autoencoder.to_json()
open('autoencoder.json', 'w+').write(json_string)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
#%%
autoencoder.fit(X_train, X_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_validation, X_validation))
test_validation = autoencoder.predict(X_validation)     
test_train = autoencoder.predict(X_train)       
 
encoder.save_weights('ConvNet_pretrained_weights.h5')
autoencoder.save_weights('autoencoder_weights.h5')
#%%

plt.figure()
plt.subplot(2,1,1)
plt.imshow(X_validation[10,0,:,:])
plt.subplot(2,1,2)
plt.imshow(test_validation[10,0,:,:])

plt.figure()
plt.subplot(2,1,1)
plt.imshow(X_train[20,0,:,:])
plt.subplot(2,1,2)
plt.imshow(test_train[20,0,:,:])


            
            
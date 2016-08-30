# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:00:03 2016

@author: charles
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:12:59 2016

@author: charles
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:35:11 2016

@author: Charles Le Losq
"""
import sys
import numpy as np

from keras.models import Sequential, Model
#from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, Dense, Activation, Lambda, merge, Dropout, Convolution1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l1, l1l2, activity_l1l2

from keras import backend as K

from sklearn import cross_validation, preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt

import silicat
import scipy

#%%
# Function definition
def tvf_th(x):
    return x[0] + x[1] / (x[3]-x[2])
#%%
###############################################################################
# Load datas
user_wants = np.genfromtxt("./inputs/wanted.csv",delimiter=',',skip_header=1)

scaler_x_g, scaler_y_g, x_i_g, y_i_g, x_t_g, y_t_g, x_v_g, y_v_g = joblib.load("./data/viscosity/saved_subsets/general_datas_sk.pkl")  
scaler_x_l, scaler_y_l, x_i_l, y_i_l, x_t_l, y_t_l, x_v_l, y_v_l = joblib.load("./data/viscosity/saved_subsets/local_datas_sk.pkl")  

X_input_train = scaler_x_g.transform(x_i_g)
Y_input_train = scaler_y_g.transform(y_i_g)

X_input_test = scaler_x_g.transform(x_t_g)
Y_input_test = scaler_y_g.transform(y_t_g)

X_input_valid = scaler_x_g.transform(x_v_g)
Y_input_valid = scaler_y_g.transform(y_v_g)

X_input_wanted = scaler_x_g.transform(user_wants[:,0:15])

#%%
def viscosity(x1,x2,x3):
    return x1 + x2/x3
    
def divis(inputs):
    return inputs[0]/inputs[1]
    
chem_input = Input(shape=(14,), name = 'chem_in')
# first 2 layers
x1 = Dense(10, W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='relu', init='he_normal')(chem_input)
A_out = Dense(1,  W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='linear', init='he_normal')(x1)
B_out = Dense(1,  W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='linear', init='he_normal')(x1)
C_out = Dense(1,  W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='linear', init='he_normal')(x1)

# Add temperature input and a custom layer
temperature_input = Input(shape=(1,), name='temp_input')
C_T_Merge = merge([C_out, temperature_input], mode='concat')
C_T_out = Dense(1,W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='linear', init='he_normal')(C_T_Merge)

B_C_T_out = merge([B_out, C_T_out], mode='concat')

output = Lambda(lambda (x1,x2): x1/x2)([B_out,C_T_out])

model = Model(input=[chem_input, temperature_input], output=[output])

model.compile(loss='mean_squared_error', optimizer='Nadam')

early_stopping = EarlyStopping(monitor='val_loss', patience=25)
model.fit([X_input_train[:,0:14],X_input_train[:,14]], Y_input_train, nb_epoch=2000, batch_size=150, validation_data=([X_input_test[:,0:14],X_input_test[:,14]], Y_input_test),callbacks=[early_stopping])

#%%
yp_train_nn = scaler_y_g.inverse_transform(model.predict([X_input_train[:,0:14],X_input_train[:,14]]))
yp_test_nn = scaler_y_g.inverse_transform(model.predict([X_input_test[:,0:14],X_input_test[:,14]]))
yp_valid_nn = scaler_y_g.inverse_transform(model.predict([X_input_valid[:,0:14],X_input_valid[:,14]]))
yp_wanted_nn = scaler_y_g.inverse_transform(model.predict([X_input_wanted[:,0:14],X_input_wanted[:,14]]))

MSE_train_nn = np.sqrt(1./float(len(yp_train_nn)-1)*np.sum((yp_train_nn[:]-y_i_g[:])**2))
MSE_test_nn = np.sqrt(1./float(len(yp_test_nn)-1)*np.sum((yp_test_nn[:]-y_t_g[:])**2))
MSE_valid_nn = np.sqrt(1./float(len(yp_valid_nn)-1)*np.sum((yp_valid_nn[:]-y_v_g[:])**2))

print('The training, testing and validating errors for the Neural Network model are: '+str(round(MSE_train_nn,2))+', '+str(round(MSE_test_nn,2))+', '+str(round(MSE_valid_nn,2))+'\n')
silicat.ml.plot_model((y_i_g,yp_train_nn),(y_t_g,yp_test_nn),(y_v_g,yp_valid_nn),(user_wants[:,14],yp_wanted_nn),plot_title="Neural Network model")

plt.figure()
plt.plot(model.history.history['loss'],'k-')
plt.plot(model.history.history['val_loss'],'b-')
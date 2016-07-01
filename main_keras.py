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

import numpy as np

import theano
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV

import matplotlib
from matplotlib import pyplot as plt

from silicat.data_import import general_input,local_input
#%%
###############################################################################
# Generate sample data

user_wants = np.genfromtxt("./inputs/wanted.csv",delimiter=',',skip_header=1)

selection = np.array([[user_wants[0,0],10.], 
                     [user_wants[0,1],10.],
                     [user_wants[0,2],10.],
                        [user_wants[0,3],10.],
                        [user_wants[0,4],10.],
                        [user_wants[0,5],10.],
                        [user_wants[0,6],10.],
                        [user_wants[0,7],10.],
                        [user_wants[0,8],10.],
                        [user_wants[0,9],10.],
                        [user_wants[0,10],10.],
                        [user_wants[0,11],10.],
                        [user_wants[0,12],10.],
                        [user_wants[0,13],10.]])
                        
scaler_x_g, scaler_y_g, x_i_g, y_i_g, x_t_g, y_t_g, x_v_g, y_v_g = general_input("./data/viscosity/viscosity.sqlite")  
scaler_x_l, scaler_y_l, x_i_l, y_i_l, x_t_l, y_t_l, x_v_l, y_v_l = local_input("./data/viscosity/viscosity.sqlite",selection)

train_g  = np.concatenate((x_i_g,y_i_g),axis=1)
test_g  = np.concatenate((x_t_g,y_t_g),axis=1)
valid_g  = np.concatenate((x_v_g,y_v_g),axis=1)

#np.savetxt("./silicat/temp/train_g.csv",train_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
#np.savetxt("./silicat/temp/test_g.csv",test_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
#np.savetxt("./silicat/temp/valid_g.csv",valid_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')

train_l  = np.concatenate((x_i_l,y_i_l),axis=1)
test_l  = np.concatenate((x_t_l,y_t_l),axis=1)
valid_l  = np.concatenate((x_v_l,y_v_l),axis=1)

#np.savetxt("./silicat/temp/train_l.csv",train_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
#np.savetxt("./silicat/temp/test_l.csv",test_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
#np.savetxt("./silicat/temp/valid_l.csv",valid_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')

#%% MODEL TRAINING

X_input_train = scaler_x_g.transform(x_i_g)
Y_input_train = scaler_y_g.transform(y_i_g)

X_input_test = scaler_x_g.transform(x_t_g)
Y_input_test = scaler_y_g.transform(y_t_g)

X_input_valid = scaler_x_g.transform(x_v_g)
X_input_wanted = scaler_x_g.transform(user_wants[:,0:15])

model = Sequential()
# 15 inputs, 10 neurons in 1 hidden layer, with tanh activation and dropout
model.add(Dense(10, init='he_normal', input_shape=(15,))) 
model.add(Activation('relu'))
model.add(Dropout(0.1))
# SECOND LAYER
model.add(Dense(2, init='he_normal')) 
model.add(Activation('relu'))
# 1 output, linear activation
model.add(Dense(1, init='he_normal'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='Nadam')


early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_input_train, Y_input_train,
          nb_epoch=2000, batch_size=150,
          validation_data=(X_input_test, Y_input_test))#,
          #callbacks=[early_stopping])

#%% MODEL OUTPUTS
          
yp_traing = scaler_y_g.inverse_transform(model.predict(X_input_train))
yp_testg = scaler_y_g.inverse_transform(model.predict(X_input_test))
yp_validg = scaler_y_g.inverse_transform(model.predict(X_input_valid))
yp_wantedg = scaler_y_g.inverse_transform(model.predict(X_input_wanted))

MSE_train_global = np.sqrt(1./float(len(yp_traing)-1)*np.sum((yp_traing[:,0]-y_i_g[:,0])**2))
MSE_test_global = np.sqrt(1./float(len(yp_testg)-1)*np.sum((yp_testg[:,0]-y_t_g[:,0])**2))
MSE_valid_global = np.sqrt(1./float(len(yp_validg)-1)*np.sum((yp_validg[:,0]-y_v_g[:,0])**2))

fig = plt.figure()
ax1 = plt.subplot(2,2,1)
#ax1.plot(y_i_l,y_train_local_pred,'ko')
ax1.plot(y_i_g,yp_traing,'bx')

ax2 = plt.subplot(2,2,2)
#ax2.plot(y_t_l,y_test_local_pred,'ko')
ax2.plot(y_t_g,yp_testg,'bx')

ax3 = plt.subplot(2,2,3)
#ax3.plot(y_v_l,y_valid_local_pred,'ko')
ax3.plot(y_v_g,yp_validg,'bx')

ax4=plt.subplot(2,2,4)
#ax4.plot(10000/user_wants[:,14],y_wanted_local_pred,'xr')
ax4.plot(10000/user_wants[:,14],yp_wantedg,'xg')
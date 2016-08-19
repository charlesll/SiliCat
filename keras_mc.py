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
from keras.layers import Input, Dense, Activation, Lambda, merge, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from sklearn.externals import joblib

from keras.regularizers import l1, l1l2, activity_l1l2

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

#%% MODEL DEFINITION
def make_model(x1_size,x2_size,x4_size,x1_l1,x2_l1,x4_l1,x1_l2,x2_l2,x4_l2,x1_acti,x2_acti,x4_acti):
    chem_input = Input(shape=(14,), name = 'chem_in')

    # first 2 layers
    x1 = Dense(x1_size, W_regularizer=l1l2(l1=x1_l1,l2=x1_l2), activation=x1_acti, init='he_normal')(chem_input)
    x2 = Dense(x2_size,  W_regularizer=l1l2(l1=x2_l1,l2=x2_l2), activation=x2_acti, init='he_normal')(x1)

    # Add temperature input and a custom layer
    temperature_input = Input(shape=(1,), name='temp_input')
    x3 = merge([x2, temperature_input], mode='concat')

    # add custom output layer
    x4 = Dense(x4_size, W_regularizer=l1l2(l1=x4_l1,l2=x4_l2), activation=x4_acti, init='he_normal')(x3)
    output = Dense(1, activation='linear', init='he_normal')(x4)
    #output = Lambda(tvf_th, output_shape=(1,))(x3)
    #model.add(Dense(1, init='he_normal'))

    model = Model(input=[chem_input, temperature_input], output=[output])

    model.compile(loss='mse', optimizer='Nadam')

    return model
    
layer1_acti = ('tanh',)
layer2_acti = ('linear',)
layer3_acti = ('tanh',)
    
model = make_model(20,3,5,0.001,0.001,0.00001,0.01,0.01,0.0001,layer1_acti[0],layer2_acti[0],layer3_acti[0])
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
# you can group inputs as [x1,x2]
model.fit([X_input_train[:,0:14],X_input_train[:,14]], Y_input_train, nb_epoch=2000, batch_size=100, validation_data=([X_input_test[:,0:14],X_input_test[:,14]], Y_input_test),callbacks=[early_stopping])
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


#%% Loop for calculating mean ESE for several training
nb_it = 20
record = np.zeros((nb_it,3))

for i in range(20):
    model.fit([X_input_train[:,0:14],X_input_train[:,14]], Y_input_train, nb_epoch=2000, batch_size=100, validation_data=([X_input_test[:,0:14],X_input_test[:,14]], Y_input_test),callbacks=[early_stopping])
    yp_train_nn = scaler_y_g.inverse_transform(model.predict([X_input_train[:,0:14],X_input_train[:,14]]))
    yp_test_nn = scaler_y_g.inverse_transform(model.predict([X_input_test[:,0:14],X_input_test[:,14]]))
    yp_valid_nn = scaler_y_g.inverse_transform(model.predict([X_input_valid[:,0:14],X_input_valid[:,14]]))
    yp_wanted_nn = scaler_y_g.inverse_transform(model.predict([X_input_wanted[:,0:14],X_input_wanted[:,14]]))

    record[i,0] = np.sqrt(1./float(len(yp_train_nn)-1)*np.sum((yp_train_nn[:]-y_i_g[:])**2))
    record[i,1] = np.sqrt(1./float(len(yp_test_nn)-1)*np.sum((yp_test_nn[:]-y_t_g[:])**2))
    record[i,2] = np.sqrt(1./float(len(yp_valid_nn)-1)*np.sum((yp_valid_nn[:]-y_v_g[:])**2))
#%%
print('The mean training, testing and validating errors are: '+str(round(np.mean(record[:,0]),2))+', '+str(round(np.mean(record[:,1]),2))+', '+str(round(np.mean(record[:,2]),2))+'\n')

#%% Now a random search in for loops
#nb_it = 10
#record = np.zeros((nb_it,3))
#record_best_params = [0,0,0,0,0,0,0,0,0,0,0,0]
#record_best_ese = np.zeros(3)+1000000000.
#
##parameter values in tuples
#layer1_units = (6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
#layer2_units = (1,2,3,4,5,6,7,8,9,10)
#layer3_units = (1,2,3,4,5,6,7,8,9,10)
#
#layer1_l1 = (0.1,0.01,0.001,0.0001,0.00001,0.000001)
#layer2_l1 = (0.1,0.01,0.001,0.0001,0.00001,0.000001)
#layer3_l1 = (0.1,0.01,0.001,0.0001,0.00001,0.000001)
#
#layer1_l2 = (0.1,0.01,0.001,0.0001,0.00001)
#layer2_l2 = (0.1,0.01,0.001,0.0001,0.00001)
#layer3_l2 = (0.1,0.01,0.001,0.0001,0.00001,0.000001)
#
#layer1_acti = ('relu','tanh')
#layer2_acti = ('relu','linear')
#layer3_acti = ('relu','tanh')
#
#total_iterations = len(layer1_units)+len(layer2_units)+len(layer3_units)+len(layer1_l1)+len(layer2_l1)+len(layer3_l1)+len(layer1_l2)+len(layer2_l2)+len(layer3_l2)+len(layer1_acti)+len(layer2_acti)+len(layer3_acti)
#iteration = 0
#
#for i in range(len(layer1_units)):
#    for j in range(len(layer2_units)):
#        for k in range(len(layer3_units)):
#            for l in range(len(layer1_l1)):
#                for m in range(len(layer2_l1)):
#                    for n in range(len(layer3_l1)):
#                        for o in range(len(layer1_l2)):
#                            for p in range(len(layer2_l2)):
#                                for q in range(len(layer3_l2)):
#                                    for r in range(len(layer1_acti)):
#                                        for s in range(len(layer2_acti)):
#                                            for t in range(len(layer3_acti)):
#                                                for it in range(nb_it):
#                                                    model = make_model(layer1_units[i],layer2_units[j],layer3_units[k],layer1_l1[l],layer2_l1[m],layer3_l1[n],layer1_l2[o],layer2_l2[p],layer3_l2[q],layer1_acti[r],layer2_acti[s],layer3_acti[t])
#                                                    
#                                                    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
#
#                                                    model.fit([X_input_train[:,0:14],X_input_train[:,14]], Y_input_train, nb_epoch=2000, batch_size=100, validation_data=([X_input_test[:,0:14],X_input_test[:,14]], Y_input_test),callbacks=[early_stopping])
#                                                    
#                                                    yp_train_nn = scaler_y_g.inverse_transform(model.predict([X_input_train[:,0:14],X_input_train[:,14]]))
#                                                    yp_test_nn = scaler_y_g.inverse_transform(model.predict([X_input_test[:,0:14],X_input_test[:,14]]))
#                                                    yp_valid_nn = scaler_y_g.inverse_transform(model.predict([X_input_valid[:,0:14],X_input_valid[:,14]]))
#                                                    
#                                                    record[i,0] = np.sqrt(1./float(len(yp_train_nn)-1)*np.sum((yp_train_nn[:]-y_i_g[:])**2))
#                                                    record[i,1] = np.sqrt(1./float(len(yp_test_nn)-1)*np.sum((yp_test_nn[:]-y_t_g[:])**2))
#                                                    record[i,2] = np.sqrt(1./float(len(yp_valid_nn)-1)*np.sum((yp_valid_nn[:]-y_v_g[:])**2))
#                                                    
#                                                mse_it = np.mean(record,0)
#                                                if (mse_it[1]+mse_it[2]) < (record_best_ese[1]+record_best_ese[2]):
#                                                    record_best_params = [layer1_units[i],layer2_units[j],layer3_units[k],layer1_l1[l],layer2_l1[m],layer3_l1[n],layer1_l2[o],layer2_l2[p],layer3_l2[q],layer1_acti[r],layer2_acti[s],layer3_acti[t]]
#                                                    record_best_ese = mse_it
#                                                print('iteration'+str(iteration)+'/'+str(total_iterations)+'\n')
#                                                iteration = iteration + 1

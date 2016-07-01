# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:50:33 2016

@author: Charles Le Losq
"""

import numpy as np
import h2o
from sklearn.svm import SVR

import matplotlib
from matplotlib import pyplot as plt

from silicat.data_import import general_input,local_input
from silicat.deep_learning import h2o_dl_global, h2o_dl_local
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

np.savetxt("./silicat/temp/train_g.csv",train_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
np.savetxt("./silicat/temp/test_g.csv",test_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
np.savetxt("./silicat/temp/valid_g.csv",valid_g,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')

train_l  = np.concatenate((x_i_l,y_i_l),axis=1)
test_l  = np.concatenate((x_t_l,y_t_l),axis=1)
valid_l  = np.concatenate((x_v_l,y_v_l),axis=1)

np.savetxt("./silicat/temp/train_l.csv",train_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
np.savetxt("./silicat/temp/test_l.csv",test_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')
np.savetxt("./silicat/temp/valid_l.csv",valid_l,delimiter=',',header='sio2,tio2,al2o3,feot,mno,bao,sro,mgo,cao,li2o,na2o,k2o,p2o5,h2o,T,viscosity')

#%%
h2o.init() # Start H2O on your local machine
h2o_dl_global(train_g,test_g,valid_g,wanted=user_wants)
h2o_dl_local(train_l,test_l,valid_l,wanted=user_wants)

y_train_local_pred = np.genfromtxt("./silicat/temp/y_train_local_pred.csv",delimiter=',',skip_header = 1)
y_test_local_pred = np.genfromtxt("./silicat/temp/y_test_local_pred.csv",delimiter=',',skip_header = 1)
y_valid_local_pred = np.genfromtxt("./silicat/temp/y_valid_local_pred.csv",delimiter=',',skip_header = 1)
y_wanted_local_pred = np.genfromtxt("./silicat/temp/y_wanted_local_pred.csv",delimiter=',',skip_header = 1)

y_train_global_pred = np.genfromtxt("./silicat/temp/y_train_global_pred.csv",delimiter=',',skip_header = 1)
y_test_global_pred = np.genfromtxt("./silicat/temp/y_test_global_pred.csv",delimiter=',',skip_header = 1)
y_valid_global_pred = np.genfromtxt("./silicat/temp/y_valid_global_pred.csv",delimiter=',',skip_header = 1)
y_wanted_global_pred = np.genfromtxt("./silicat/temp/y_wanted_global_pred.csv",delimiter=',',skip_header = 1)

MSE_train_local = np.sqrt(1./float(len(y_train_local_pred)-1)*np.sum((y_train_local_pred[:]-y_i_l[:,0])**2))
MSE_test_local = np.sqrt(1./float(len(y_test_local_pred)-1)*np.sum((y_test_local_pred[:]-y_t_l[:,0])**2))
MSE_valid_local = np.sqrt(1./float(len(y_valid_local_pred)-1)*np.sum((y_valid_local_pred[:]-y_v_l[:,0])**2))

MSE_train_global = np.sqrt(1./float(len(y_train_global_pred)-1)*np.sum((y_train_global_pred[:]-y_i_g[:,0])**2))
MSE_test_global = np.sqrt(1./float(len(y_test_global_pred)-1)*np.sum((y_test_global_pred[:]-y_t_g[:,0])**2))
MSE_valid_global = np.sqrt(1./float(len(y_valid_global_pred)-1)*np.sum((y_valid_global_pred[:]-y_v_g[:,0])**2))

#%%
###############################################################################
# Plotting

fig = plt.figure()
ax1 = plt.subplot(2,2,1)
ax1.plot(y_i_l,y_train_local_pred,'ko')
ax1.plot(y_i_g,y_train_global_pred,'bx')

ax2 = plt.subplot(2,2,2)
ax2.plot(y_t_l,y_test_local_pred,'ko')
ax2.plot(y_t_g,y_test_global_pred,'bx')

ax3 = plt.subplot(2,2,3)
ax3.plot(y_v_l,y_valid_local_pred,'ko')
ax3.plot(y_v_g,y_valid_global_pred,'bx')

ax4=plt.subplot(2,2,4)
ax4.plot(10000/user_wants[:,14],y_wanted_local_pred,'xr')
ax4.plot(10000/user_wants[:,14],y_wanted_global_pred,'xg')




# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:50:33 2016

@author: Charles Le Losq
"""

import numpy as np
import h2o
from sklearn.externals import joblib

from matplotlib import pyplot as plt

from silicat.h2o_ml import h2o_dl_global, h2o_dl_local
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
                        
train_g, test_g, valid_g = joblib.load("./data/viscosity/saved_subsets/general_datas_h2o.pkl")  
train_l, test_l, valid_l = joblib.load("./data/viscosity/saved_subsets/local_datas_h2o.pkl")   
#%%
h2o.init() # Start H2O on your local machine
h2o_dl_global(train_g,test_g,valid_g,wanted=user_wants)
h2o_dl_local(train_l,test_l,valid_l,wanted=user_wants)

#%%
y_train_local_pred = np.genfromtxt("./silicat/temp/y_train_local_pred.csv",delimiter=',',skip_header = 1)
y_test_local_pred = np.genfromtxt("./silicat/temp/y_test_local_pred.csv",delimiter=',',skip_header = 1)
y_valid_local_pred = np.genfromtxt("./silicat/temp/y_valid_local_pred.csv",delimiter=',',skip_header = 1)
y_wanted_local_pred = np.genfromtxt("./silicat/temp/y_wanted_local_pred.csv",delimiter=',',skip_header = 1)

y_train_global_pred = np.genfromtxt("./silicat/temp/y_train_global_pred.csv",delimiter=',',skip_header = 1)
y_test_global_pred = np.genfromtxt("./silicat/temp/y_test_global_pred.csv",delimiter=',',skip_header = 1)
y_valid_global_pred = np.genfromtxt("./silicat/temp/y_valid_global_pred.csv",delimiter=',',skip_header = 1)
y_wanted_global_pred = np.genfromtxt("./silicat/temp/y_wanted_global_pred.csv",delimiter=',',skip_header = 1)

MSE_train_local = np.sqrt(1./float(len(y_train_local_pred)-1)*np.sum((y_train_local_pred[:]-train_l[:,15])**2))
MSE_test_local = np.sqrt(1./float(len(y_test_local_pred)-1)*np.sum((y_test_local_pred[:]-test_l[:,15])**2))
MSE_valid_local = np.sqrt(1./float(len(y_valid_local_pred)-1)*np.sum((y_valid_local_pred[:]-valid_l[:,15])**2))

MSE_train_global = np.sqrt(1./float(len(y_train_global_pred)-1)*np.sum((y_train_global_pred[:]-train_g[:,15])**2))
MSE_test_global = np.sqrt(1./float(len(y_test_global_pred)-1)*np.sum((y_test_global_pred[:]-test_g[:,15])**2))
MSE_valid_global = np.sqrt(1./float(len(y_valid_global_pred)-1)*np.sum((y_valid_global_pred[:]-valid_g[:,15])**2))

#%%
###############################################################################
# Plotting

fig = plt.figure()
ax1 = plt.subplot(2,2,1)
ax1.plot(train_l[:,15],y_train_local_pred,'ko')
ax1.plot(train_g[:,15],y_train_global_pred,'bx')

ax2 = plt.subplot(2,2,2)
ax2.plot(test_l[:,15],y_test_local_pred,'ko')
ax2.plot(test_g[:,15],y_test_global_pred,'bx')

ax3 = plt.subplot(2,2,3)
ax3.plot(valid_l[:,15],y_valid_local_pred,'ko')
ax3.plot(valid_g[:,15],y_valid_global_pred,'bx')

ax4=plt.subplot(2,2,4)
ax4.plot(10000/user_wants[:,14],y_wanted_local_pred,'xr')
ax4.plot(10000/user_wants[:,14],y_wanted_global_pred,'xg')




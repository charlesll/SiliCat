# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:35:11 2016

@author: Charles Le Losq
"""

import numpy as np
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

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
# First we create a KFold cross validation iterator that will be used in GridSearchCV
kf = cross_validation.KFold(len(x_i_g), n_folds=10, shuffle=True, random_state=None)
#cv = cross_validation.ShuffleSplit(len(X_train), n_iter=20, test_size=0.60, random_state=0)

# Then we do a grid search cross validated
mg = GridSearchCV(SVR(kernel='rbf'), cv=kf,param_grid={"C":[50,75,100,150,200],"gamma": [0.01,0.1,0.2,0.4]},n_jobs = 1)

# We fit data
mg.fit(scaler_x_g.transform(x_i_g),scaler_y_g.transform(y_i_g.ravel()))

# Printing best params
print(mg.best_params_)
print()

y_train_global_pred = scalery.inverse_transform(mg.predict(scaler_x_g.transform(x_i_g)))
y_test_global_pred = scalery.inverse_transform(mg.predict(scaler_x_g.transform(x_t_g)))
y_valid_global_pred = scalery.inverse_transform(mg.predict(scaler_x_g.transform(x_v_g)))
#y_wanted_local_pred = mg.predict(scaler_x_g(user_wants[0,1:14]))


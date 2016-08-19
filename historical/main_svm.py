# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:35:11 2016

@author: Charles Le Losq
"""

import numpy as np
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt

from silicat.data_import import general_input,local_input
import silicat
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
# First we create a KFold cross validation iterator that will be used in GridSearchCV
silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'svm_global')
mg = silicat.ml.load_model('svm_global',origin="Scikit")

#%%
# Printing best params
print('Best parameters for global model:\n')
print(mg.best_params_)
print('Best parameters for global model:\n')
print(mg.best_params_)
print()

yp_traing = scaler_y_g.inverse_transform(mg.predict(X_input_train))
yp_testg = scaler_y_g.inverse_transform(mg.predict(X_input_test))
yp_validg = scaler_y_g.inverse_transform(mg.predict(X_input_valid))
yp_wantedg = scaler_y_g.inverse_transform(mg.predict(X_input_wanted))


MSE_train_global = np.sqrt(1./float(len(yp_traing)-1)*np.sum((yp_traing[:]-y_i_g[:,0])**2))
MSE_test_global = np.sqrt(1./float(len(yp_testg)-1)*np.sum((yp_testg[:]-y_t_g[:,0])**2))
MSE_valid_global = np.sqrt(1./float(len(yp_validg)-1)*np.sum((yp_validg[:]-y_v_g[:,0])**2))

silicat.ml.plot_model((y_i_g,yp_traing),(y_t_g,yp_testg),(y_v_g,yp_validg),(user_wants[:,14],yp_wantedg),plot_title="My SVM Model")



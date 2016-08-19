# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:20:20 2015

@author: closq
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:15:00 2015

@author: closq
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:46:16 2015

@author: closq
"""
import time
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
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
#TRAINING                        
silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'gbr_global')
mg = silicat.ml.load_model('gbr_global',origin="Scikit")

#%%
# Printing best params
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

silicat.ml.plot_model((y_i_g,yp_traing),(y_t_g,yp_testg),(y_v_g,yp_validg),(user_wants[:,14],yp_wantedg),plot_title="My GBR Model")






#%%



### SAVING STUFFS

################################################################################
## look at the results
#plt.figure(1)
#gs = gridspec.GridSpec(1, 2)
#ax1 = plt.subplot(gs[0,0])
#ax2 = plt.subplot(gs[0,1])
#
#ax1.plot(yinput, pred_train, 'gs', label='Training')
#ax2.plot(out_test, pred_test, 'rs', label='Independent Testing')
#ax1.legend(loc="best")
#ax2.legend(loc="best")
#
################################################################################
## Plot training deviance
#
## compute test set deviance
#test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
#for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
#    test_score[i] = clf.loss_(y_test, y_pred)
#
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title('Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#         label='Training Set Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#         label='Test Set Deviance')
#plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
#plt.ylabel('Deviance')



##Manual gridsearch
## Best values 5, 1.5, 10000, 16 => mse_test = 0.617
#maxdepth = np.array([4,5,6,7,8])
#learningrate = np.array([1.0,1.5,2.0,3.0])
#numberestimators = np.array([10000,50000,100000])
#minsamplessplit = np.array([10,16,20,25])
#
##Arrays for storing the results
#ysc_train = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),len(y_train)))
#ysc_test = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),len(y_test)))
#pred_train = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),len(y_train)))
#pred_test = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),len(y_test)))
#
#mse_train = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),1))
#mse_test = np.zeros((len(maxdepth),len(learningrate),len(numberestimators),len(minsamplessplit),1))
#
#bestparams = np.zeros((4))
#nvc = len(maxdepth)*len(learningrate)*len(numberestimators)*len(minsamplessplit)
#compteur = 0
#best_mse_test = 1000
#best_mse_train = 1000
#
## Loop for gridsearch
#for i in range(len(maxdepth)):
#     # Printing the iterations
#    print("\nSearch number: "+str(compteur))
#    for j in range(len(learningrate)):
#        for k in range(len(numberestimators)):
#            for l in range(len(minsamplessplit)):
#                     
#                if compteur == 0:
#                    print("Starting the grid search for the parameters") 
#                    print("There is "+str(nvc)+" values to check")
#                    #time.sleep(1)
#                    
#                
#                params = {'n_estimators': numberestimators[k] , 'max_depth': maxdepth[i], 'min_samples_split': minsamplessplit[l], 'learning_rate': learningrate[j], 'loss': 'ls','random_state': 0}    
#
#                gbr = ensemble.GradientBoostingRegressor(**params)
#                gbr.fit(X_train,y_train)
#                ysc_train[i,j,k,l,:] = gbr.predict(X_train)
#                ysc_test[i,j,k,l,:] = gbr.predict(X_test)
#                # Un-scalling
#                pred_train[i,j,k,l,:] = scalery.inverse_transform(ysc_train[i,j,k,l,:])
#                pred_test[i,j,k,l,:] = scalery.inverse_transform(ysc_test[i,j,k,l,:])
#                
#                # MSE of the various technics
#                mse_train[i,j,k,l,0] = np.sqrt(1/(float(len(pred_train[i,j,k,l,:])))*sum((pred_train[i,j,k,l,:]-yinput[:])**2))
#                mse_test[i,j,k,l,0] = np.sqrt(1/(float(len(pred_test[i,j,k,l,:])))*sum((pred_test[i,j,k,l,:]-out_test[:])**2))
#                
#                if (mse_test[i,j,k,l,0] < best_mse_test):
#                    bestparams[:] = maxdepth[i], learningrate[j], numberestimators[k], minsamplessplit[l]
#                    best_mse_test = mse_test[i,j,k,l,0]
#                    best_mse_train = mse_train[i,j,k,l,0]
#                
#                compteur = compteur + 1
#                
                
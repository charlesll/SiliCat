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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_import import general_input

###############################################################################
# Generate sample data
x_i, y_i, x_t, y_t, x_v, y_v = general_input()

###############################################################################

# Standardization of the data, the scaler for x and y are saved for latter use
scalerx = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_i)
scalery = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(y_i)

X_train = scalerx.transform(x_i)
y_train = np.ravel(scalery.transform(y_i))

X_test = scalerx.transform(x_t)
y_test = np.ravel(scalery.transform(y_t))

X_valid = scalerx.transform(x_v)
y_valid = np.ravel(scalery.transform(y_v))

###############################################################################
# Data cross-validation
kf = cross_validation.KFold(len(X_train), n_folds=10, shuffle=True, random_state=0)
cv = cross_validation.ShuffleSplit(len(X_train), n_iter=20, test_size=0.25, random_state=0)

###############################################################################
# GRADIENT Boosting REGRESSION model

# Then we do a grid search cross validated
clf = GridSearchCV(ensemble.GradientBoostingRegressor(), cv=cv,param_grid={"max_depth": [3,4,5],"learning_rate": [0.005,0.01,0.05,0.1,0.25,0.50]},n_jobs = 1)

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
                
#clf = gbr

# We fit data
clf.fit(X_train, y_train)

# Printing best params
print(clf.best_params_)
print()

# Fit regression model
ysc_train = clf.predict(X_train)
ysc_test = clf.predict(X_test)

###############################################################################
# Un-scalling
pred_train = scalery.inverse_transform(ysc_train)
pred_test = scalery.inverse_transform(ysc_test)

###############################################################################
# MSE of the various technics
mse_train = np.sqrt(1/(float(len(pred_train)))*sum((pred_train[:]-np.array(y_i))**2))
mse_test = np.sqrt(1/(float(len(pred_test)))*sum((pred_test[:]-np.array(y_t))**2))

print("MSE TRAIN = "+str(mse_train))
print("MSE TEST = "+str(mse_test))



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
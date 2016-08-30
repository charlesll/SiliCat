# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:43:18 2016

@author: Charles Le Losq

Example of using sklearn for baseline treatment

"""

import sys, time

import numpy as np
import scipy

import matplotlib
from matplotlib import pyplot as plt

from keras.models import Sequential, Model
#from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, Dense, Activation, Lambda, merge, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l1, l1l2, activity_l1l2

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

import silicat


spectre = np.genfromtxt('../data/raman/r040.txt',skip_header=1)

x = spectre[:,0]
y = spectre[:,1]

bir = np.array([[110., 230.],[1300.,2700.],[3800.,4000.]])

for i in range(len(bir)):
    if i == 0:
        yafit = spectre[np.where((spectre[:,0]> bir[i,0]) & (spectre[:,0] < bir[i,1]))] 
    else:
        je = spectre[np.where((spectre[:,0]> bir[i,0]) & (spectre[:,0] < bir[i,1]))]
        yafit = np.concatenate((yafit,je),axis=0)


X_scaler = preprocessing.StandardScaler().fit(yafit[:,0].reshape(-1,1))
y_scaler = preprocessing.StandardScaler().fit(yafit[:,1].reshape(-1,1))

X_i = X_scaler.transform(yafit[:,0].reshape(-1,1))
y_i = y_scaler.transform(yafit[:,1].reshape(-1,1))

#%%
#clf = svm.SVR(C=10., gamma=0.1,kernel='rbf')
#clf = KernelRidge(alpha=1, kernel='rbf')

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_i, y_i[:].ravel())
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s", svr_fit)

t0 = time.time()
kr.fit(X_i, y_i[:].ravel())
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s", kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / np.size(X_i,0)
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr_sc = svr.predict(X_scaler.transform(x.reshape(-1,1)))
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s", (x.reshape(-1,1).shape[0], svr_predict))

t0 = time.time()
y_kr_sc = kr.predict(X_scaler.transform(x.reshape(-1,1)))
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s", (x.reshape(-1,1).shape[0], kr_predict))


#y_pred_sc = clf.fit(X_i, y_i[:].ravel()).predict(X_scaler.transform(x.reshape(-1,1))) 
#y_pred_sc = clf.fit(yafit[:,0].reshape(-1,1), yafit[:,1].ravel()).predict(x.reshape(-1,1)) 



#model = Sequential()
##model.add(Dense(input_dim = 1, output_dim = 120))
##model.add(Activation('tanh'))
##model.add(Dense(input_dim = 1, output_dim = 60))
##model.add(Activation('tanh'))
#model.add(Dense(input_dim = 1, output_dim = 5))
#model.add(Activation('tanh'))
##model.add(Dropout(0.1))
##model.add(Dense(input_dim = 100, output_dim = 30))
##model.add(Activation('tanh'))
##model.add(Dropout(0.5))
#model.add(Dense(input_dim = 5, output_dim = 1))
#model.add(Activation('linear'))
#model.compile(loss='mean_absolute_error', optimizer='Nadam')
#early_stopping = EarlyStopping(monitor='val_loss', patience=1000)
#model.fit([X_i], y_i, nb_epoch=2000, batch_size=150, validation_split=0.2,callbacks=[early_stopping])
#y_pred_sc = model.predict(X_scaler.transform(x.reshape(-1,1)))

y_pred_svm = y_scaler.inverse_transform(y_svr_sc)
y_pred_kr = y_scaler.inverse_transform(y_kr_sc)
#%%
plt.figure()
plt.plot(x,y,'k-')
plt.plot(yafit[:,0],yafit[:,1],'r.')
plt.plot(x,y_pred_svm,'g-')
plt.plot(x,y_pred_kr,'m-')


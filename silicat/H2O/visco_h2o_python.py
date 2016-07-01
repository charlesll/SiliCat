# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:41:36 2015

@author: closq
"""

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


###############################################################################
# Generate sample data
train = np.genfromtxt('../data/H2O_train.txt',delimiter='\t',skip_header = 1)
valid = np.genfromtxt('../data/H2O_valid.txt',delimiter='\t',skip_header = 1)
test = np.genfromtxt('../data/H2O_test.txt',delimiter='\t',skip_header = 1)

###############################################################################
# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.2)
clf.fit(train[:,0:15])
y_pred_train = clf.predict(train[:,0:15])
y_pred_test = clf.predict(valid[:,0:15])
y_pred_outliers = clf.predict(test[:,0:15])
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
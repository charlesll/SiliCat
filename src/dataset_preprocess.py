
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:17:42 2015

Shuffle and split training dataset for h2o

@author: closq
"""

import cPickle, gzip, numpy as np

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.utils import shuffle
###############################################################################
# Generate sample data
h2o_train = np.genfromtxt('../data/Train_forH2O.txt',delimiter='\t',skip_header = 1)

shuf_train = shuffle(h2o_train, random_state=0)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(shuf_train[:,0:11], shuf_train[:,12], test_size=0.25, random_state=0)

train_final = np.zeros((len(X_train),h2o_train.shape[1]))
test_final = np.zeros((len(X_train),h2o_train.shape[1]))

train_final[:,0:11] = X_train
train_final[:,12] = y_train[:]

test_final[:,0:11] = X_test
test_final[:,12] = y_test[:]

# Saving the dataset with cPickle
np.savetxt("../data/Train_forH2O_shuffled.txt",train_final)
np.savetxt("../data/Test_forH2O_shuffled.txt",test_final)
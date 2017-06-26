# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:36:40 2016

@author: charles
"""

import numpy as np
import os

import matplotlib
from matplotlib import pyplot as plt
gfont = {'fontname':'Geneva'}
afont = {'fontname':'Arial'}
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

import theano
import keras

from keras.models import Sequential, Model
#from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, Dense, Activation, Lambda, merge, Dropout
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

from keras.regularizers import l1, l1l2, activity_l1l2

from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

from silicat.data_import import general_input,local_input

def train_nn(X_input_train,Y_input_train,X_input_test,Y_input_test,name,**options):
    """
        Function to train feed-forward neural network using the Keras library, with Theano backend.
        
        This functions trains and saves the network as a pickle file in ./silicat/saved/
         
        INPUTS:
        
        X_input_train: Numpy array 
            The training input values
            
        X_input_train: Numpy array 
            The training input values

        X_input_train: Numpy array 
            The training input values
        
        X_input_train: Numpy array 
            The training input values

        name: String
            The name of the file for saving the network

        OPTIONS:

        mode: "local" or "global"
            Select if the network is for a local or global dataset, for viscosity fitting.

            We are training neural networks with different sizes when looking at local or global datasets. See docs.            
            
            default: "global"            
            
        OUTPUTS:
        
        A saved network in ./silicat/saved/ with filename = name.
        
    """
    # Default values
    if options.get("mode") == None:
        mode = "global"
    else:
        mode = options.get("mode") 
    
    try:
        if mode == "global":
            
            model = Sequential()
    
            # 1 LAYER  
    
            model.add(Dense(16, init='lecun_uniform',W_regularizer=l1l2(l1=0.001,l2=0.01), input_shape=(15,)))
            #model.add(PReLU())
            model.add(Activation('relu'))
            #model.add(Dropout(0.1))
                
            # 2 LAYER
            model.add(Dense(8, init='lecun_uniform',W_regularizer=l1l2(l1=0.0001,l2=0.001))) 
            #model.add(PReLU())            
            model.add(Activation('relu'))
        
            # 3 LAYER
            model.add(Dense(4, init='lecun_uniform',W_regularizer=l1l2(l1=0.0001,l2=0.001))) 
            #model.add(PReLU())            
            model.add(Activation('relu'))
            
            # 4 LAYER
            #model.add(Dense(100, init='he_normal',W_regularizer=l1l2(l1=0.0001,l2=0.0025))) 
            #model.add(Activation('relu'))
            
            # 5 LAYER
            #model.add(Dense(100, init='he_normal',W_regularizer=l1l2(l1=0.0001,l2=0.0025))) 
            #model.add(Activation('relu'))
        
            # 1 output, linear activation
            model.add(Dense(1, init='lecun_uniform'))
            model.add(Activation('linear'))
            
        if mode == "local":
            
            model = Sequential()
    
            # 1 LAYER
            model.add(Dense(8, init='he_normal',W_regularizer=l1l2(l1=0.001,l2=0.001), input_shape=(15,))) 
            model.add(Activation('relu'))
            #model.add(Dropout(0.1))
                
            # 2 LAYER
            model.add(Dense(2, init='he_normal',W_regularizer=l1l2(l1=0.001,l2=0.0001))) 
            model.add(Activation('relu'))
        
            # 1 output, linear activation
            model.add(Dense(1, init='he_normal'))
            model.add(Activation('linear'))
    
        if (mode != "global" and mode != "local"):
            raise NameError('NameError: mode should be set to "global" or "local"')
            
    except NameError as err:
        print(err.args)    
    
#    chem_input = Input(shape=(14,), name = 'chem_in')
#    
#    # first 2 layers
#    x1 = Dense(12, W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='relu', init='he_normal')(chem_input)
#    x2 = Dense(3,  W_regularizer=l1l2(l1=0.0001,l2=0.003), activation='relu', init='he_normal')(x1)
#    
#    # Add temperature input and a custom layer
#    temperature_input = Input(shape=(1,), name='temp_input')
#    x3 = merge([x2, temperature_input], mode='concat')
#    
#    # add custom output layer
#    x4 = Dense(4, activation='relu', init='he_normal')(x3)
#    output = Dense(1, activation='linear', init='he_normal')(x4)
#    #output = Lambda(lambda x: (x[0] + x[1] / (x[3]-x[2])), output_shape=(1,))(x3)
#    #model.add(Dense(1, init='he_normal'))
#    #model.add(Activation('linear'))
#    
#    model = Model(input=[chem_input, temperature_input], output=[output])
#    #model = Model(input=[chem_input], output=[output])
#    
    
    model.compile(loss='mse', optimizer='Nadam')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=25)
    # you can group inputs as [x1,x2]
    model.fit(X_input_train, Y_input_train, nb_epoch=2000, batch_size=100, validation_data=(X_input_test, Y_input_test),callbacks=[early_stopping])
    #model.fit([X_input_train[:,0:14],X_input_train[:,14]], Y_input_train, nb_epoch=2000, batch_size=150, validation_data=([X_input_test[:,0:14],X_input_test[:,14]], Y_input_test),callbacks=[early_stopping])
       
    json_string = model.to_json()
    open(os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.json', 'w+').write(json_string)
    model.save_weights(os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.h5',overwrite=True)
    
def train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,name):
    """
        Function to train a support vector machine model with the Scikit Learn API
        
        This functions trains and saves the network as a pickle file in ./silicat/saved/
         
        INPUTS:
        
        X_input_train: Numpy array 
            The training input values
            
        X_input_train: Numpy array 
            The training input values

        X_input_train: Numpy array 
            The training input values
        
        X_input_train: Numpy array 
            The training input values

        name: String
            The name of the file for saving the network

        OUTPUTS:
        
        A saved network in ./silicat/saved/ with filename = name.
        
    """
    # this is for the cross-validation stuff, we put the test data at the end and get their indexes
    X_dataset = np.concatenate([X_input_train,X_input_test])    
    Y_dataset = np.concatenate([Y_input_train,Y_input_test])
    
    #ttt = test[len(x_i_g):len(x_i_g)+len(x_t_g),:] /// (ttt==x_t_g).all() => I found this technic like that, playing around
    test_idx = np.arange(len(X_input_train),len(X_input_train)+len(X_input_test))
    
    # and we setup the cross-validation here
    ps = cross_validation.PredefinedSplit(test_fold=test_idx) 
    
    # now we perform the gridsearch and fit the model, get the best nmodel and save it using joblib
    #mg = GridSearchCV(SVR(kernel='rbf',epsilon=0.1), cv=10,param_grid={"C":[0.01,0.1,1.0,10.0,100.0], "epsilon"=[0]},n_jobs = 1)
    mg = SVR(kernel='rbf',C=1.5, epsilon=0.01)
    mg.fit(X_input_train,Y_input_train.ravel())
    joblib.dump(mg, os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.pkl') 
    
def train_gbr(X_input_train,Y_input_train,X_input_test,Y_input_test,name):
    """
        Function to train a Gradient Boosting regression model with the Scikit Learn API
        
        This functions trains and saves the network as a pickle file in ./silicat/saved/
         
        INPUTS:
        
        X_input_train: Numpy array 
            The training input values
            
        X_input_train: Numpy array 
            The training input values

        X_input_train: Numpy array 
            The training input values
        
        X_input_train: Numpy array 
            The training input values

        name: String
            The name of the file for saving the network

        OUTPUTS:
        
        A saved network in ./silicat/saved/ with filename = name.
        
    """
    # this is for the cross-validation stuff, we put the test data at the end and get their indexes
    X_dataset = np.concatenate([X_input_train,X_input_test])    
    Y_dataset = np.concatenate([Y_input_train,Y_input_test])
    
    #ttt = test[len(x_i_g):len(x_i_g)+len(x_t_g),:] /// (ttt==x_t_g).all() => I found this technic like that, playing around
    test_idx = np.arange(len(X_input_train),len(X_input_train)+len(X_input_test))
    
    # and we setup the cross-validation here
    ps = cross_validation.PredefinedSplit(test_fold=test_idx) 
    
    # now we perform the gridsearch and fit the model, get the best nmodel and save it using joblib
    #mg = GridSearchCV(ensemble.GradientBoostingRegressor(), cv=ps,param_grid={"max_depth": [4,5,6],"learning_rate": [0.01,0.1,1.0,10.0]},n_jobs = 1)
    # Fit regression model
    params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 1,
          'learning_rate': 0.1, 'loss': 'ls'}
    mg = ensemble.GradientBoostingRegressor(**params)
    mg.fit(X_input_train,Y_input_train.ravel())
    joblib.dump(mg, os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.pkl') 

def load_model(name, origin = None):
    """
        Function to load a model pre-trained with either Scikit learn or Keras.
        
        This functions loads the model with the filename = name previously saved as a pickle file in ./silicat/saved/
         
        INPUTS:

        name: String
            The name of the file for saving the network
            
        origin: "Scikit" or "Keras"
            The API used to train the model. (Just to handle a few differences in the loading process) 

        OUTPUTS:
        
        model: A Scikit Learn or Keras model object
        
    """
    # default value for origin
    if origin == None:
        origin == "Scikit"
        
    try:
        if origin == "Scikit":
            return joblib.load(os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.pkl')
            
        if origin == "Keras":
            model = model_from_json(open(os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.json').read())
            model.load_weights(os.path.dirname(os.path.abspath(__file__))+'/saved/'+name+'.h5')
            model.compile(loss='mse', optimizer='Nadam')
            return model
    
        if (origin != "Scikit" and origin != "Keras"):
            raise NameError('NameError: origin should be set to Scikit or Keras.')
            
    except NameError as err:
        print(err.args)

def plot_model(train_Ys,test_Ys,valid_Ys,prediction_Ys,plot_title=None):
    # default value for origin
    if plot_title == None:
        plot_title == "My model plot"
        
    fig = plt.figure()
    plt.title(plot_title)
    
    ax1 = plt.subplot(2,2,1)
    ax1.plot(train_Ys[0],train_Ys[1],'b.', label='Training set')
    ax1.set_xlabel('Measured viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax1.set_ylabel('Calculated viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax1.set_title('Training set',fontsize = 14, fontweight = 'bold',**afont)
    ax1.set_xlim([0,20])
    ax1.set_ylim([0,20])
    
    ax2 = plt.subplot(2,2,2)
    ax2.plot(test_Ys[0],test_Ys[1],'c.', label='Testing set')
    ax2.set_xlabel('Measured viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax2.set_ylabel('Calculated viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax2.set_title('Testing set',fontsize = 14, fontweight = 'bold',**afont)
    ax2.set_xlim([0,20])
    ax2.set_ylim([0,20])
    
    ax3 = plt.subplot(2,2,3)
    ax3.plot(valid_Ys[0],valid_Ys[1],'m.', label='Validation set')
    ax3.set_xlabel('Measured viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax3.set_ylabel('Calculated viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax3.set_title('Validation set',fontsize = 14, fontweight = 'bold',**afont)
    ax3.set_xlim([0,20])
    ax3.set_ylim([0,20])
    
    ax4=plt.subplot(2,2,4)
    ax4.plot(prediction_Ys[0]*10000,prediction_Ys[1],'xg', label='User wanted prediction')
    ax4.set_xlabel('10$^{4}$/T, K$^{-1}$',fontsize = 14, fontweight = 'bold',**afont)
    ax4.set_ylabel('Predicted viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax4.set_title('Desired predictions',fontsize = 14, fontweight = 'bold',**afont)
    
    plt.tight_layout()
    plt.show()
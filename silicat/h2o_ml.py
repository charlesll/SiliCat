# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:27:35 2016

@author: Charles Le Losq
"""

import h2o
import os
import numpy as np
import csv

def h2o_dl_global(train,test,valid,**options):
    """
        This is a function to automate the run of the h2o deep learning estimator.
        This function is aimed at fitting the entire dataset. Weights and bias will be outputed in /data/viscosity/best_dl_model
        
        INPUTS:
        
        train: Numpy array
            Contains the training dataset, standardized with sklearn
            
        test: Numpy array
            Contains the testing dataset, standardized with sklearn
            
        valid: Numpy array
            Contains the validation dataset, standardized with sklearn. h2o_deeplearning will not see this dataset during the training process
            
        OUTPUTS:
            weights and biases written in /data/viscosity/best_dl_model
    """
    
    # generate h2o data frames
    df_train = h2o.H2OFrame.from_python(train.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    df_test = h2o.H2OFrame.from_python(test.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    df_valid = h2o.H2OFrame.from_python(valid.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])

    #generate the estimator object
    dl_estimator = h2o.estimators.deeplearning.H2ODeepLearningEstimator(
     standardize=True, # 
     activation = "Tanh", 
	hidden = [16,8,4],
	#input_dropout_ratio = .1,
	#hidden_dropout_ratios = .1,
	epochs = 10000, #3000 
	variable_importances = True,
	use_all_factor_levels = True,
	train_samples_per_iteration = -2,
	adaptive_rate = True,
	l1 = 0.0001, 
	l2 = 0.0025, 
	shuffle_training_data = False, 
	reproducible = False,
	loss = "Automatic",
	score_interval = 5,
	score_training_samples = 10000,
	score_validation_samples = 0,
	score_duty_cycle = 0.1,
	overwrite_with_best_model = True,
	#seed = 7979706519224486000,
	rho = 0.995,
	epsilon = 1e-8,
	initial_weight_distribution = "UniformAdaptive",
	regression_stop  = 0.000001, 
	diagnostics = True,
	fast_mode = False,
	force_load_balance = True,
	single_node_mode = False,
	quiet_mode = False,
	sparse = False,
	col_major = False,
	average_activation = 0,
	sparsity_beta = 0,
      export_weights_and_biases = True)
 
    # run the model training
    dl_estimator.train(x=range(15),y=15,training_frame=df_train,validation_frame = df_test)
    
    y_train_pred = dl_estimator.predict(df_train)
    y_test_pred = dl_estimator.predict(df_test)
    y_valid_pred = dl_estimator.predict(df_valid)
    
    dl_estimator.model_performance(df_train)
    dl_estimator.model_performance(df_test)
    dl_estimator.model_performance(df_valid)

    #examples for outputs the biases and weights
    #weights0 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.weights(dl_estimator,0)
    #weights1 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.weights(dl_estimator,1)
    #weights2 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.weights(dl_estimator,2)
    #weights3 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.weights(dl_estimator,3)

    #bias0 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.biases(dl_estimator,0)
    #bias1 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.biases(dl_estimator,1)
    #bias2 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.biases(dl_estimator,2)
    #bias3 = h2o.estimators.deeplearning.H2ODeepLearningEstimator.biases(dl_estimator,3)
    
    #h2o.export_file(weights0, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/w0.csv', force=True)
    #h2o.export_file(weights1, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/w1.csv', force=True)    
    #h2o.export_file(weights2, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/w2.csv', force=True)

    #h2o.export_file(bias0, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/b0.csv', force=True)
    #h2o.export_file(bias1, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/b1.csv', force=True)
    #h2o.export_file(bias2, os.path.dirname(os.path.abspath(__file__))+'/../data/viscosity/best_dl_wb/b2.csv', force=True)
    
    h2o.export_file(y_train_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_train_global_pred.csv', force=True)
    h2o.export_file(y_test_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_test_global_pred.csv', force=True)
    h2o.export_file(y_valid_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_valid_global_pred.csv', force=True)
    
    # Default values
    if options.get("wanted") == None:
        wanted = None
    else:
        wanted = options.get("wanted")
        df_wanted = h2o.H2OFrame.from_python(wanted.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
        y_wanted_pred = dl_estimator.predict(df_wanted)
        h2o.export_file(y_wanted_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_wanted_global_pred.csv', force=True)
        
def h2o_dl_local(train,test,valid,**options):
    """
        This is a function to automate the run of the h2o deep learning estimator.
        This function is aimed at fitting the entire dataset. Weights and bias will be outputed in /data/viscosity/best_dl_model
        
        INPUTS:
        
        train: Numpy array
            Contains the training dataset, standardized with sklearn
            
        test: Numpy array
            Contains the testing dataset, standardized with sklearn
            
        valid: Numpy array
            Contains the validation dataset, standardized with sklearn. h2o_deeplearning will not see this dataset during the training process
            
        OUTPUTS:
            weights and biases written in /data/viscosity/best_dl_model
    """

    # generate h2o data frames
    df_train = h2o.H2OFrame.from_python(train.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    df_test = h2o.H2OFrame.from_python(test.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    df_valid = h2o.H2OFrame.from_python(valid.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    
    #generate the estimator object
    dl_estimator = h2o.estimators.deeplearning.H2ODeepLearningEstimator(
     standardize=True, # 
     activation = "Tanh", 
	hidden = [3],
	#input_dropout_ratio = .1,
	#hidden_dropout_ratios = .1,
	epochs = 10000, #3000 
	variable_importances = True,
	use_all_factor_levels = True,
	train_samples_per_iteration = -2,
	adaptive_rate = True,
	l1 = 0.0001, 
	l2 = 0.002, 
	shuffle_training_data = False, 
	reproducible = True,
	loss = "Automatic",
	score_interval = 5,
	score_training_samples = 10000,
	score_validation_samples = 0,
	score_duty_cycle = 0.1,
	overwrite_with_best_model = True,
	#seed = 7979706519224486000,
	rho = 0.995,
	epsilon = 1e-8,
	initial_weight_distribution = "UniformAdaptive",
	regression_stop  = 0.000001, 
	diagnostics = True,
	fast_mode = False,
	force_load_balance = True,
	single_node_mode = False,
	quiet_mode = False,
	sparse = False,
	col_major = False,
	average_activation = 0,
	sparsity_beta = 0,
      export_weights_and_biases = True)
 
    # run the model training
    dl_estimator.train(x=range(15),y=15,training_frame=df_train,validation_frame = df_test)
    
    y_train_pred = dl_estimator.predict(df_train)
    y_test_pred = dl_estimator.predict(df_test)
    y_valid_pred = dl_estimator.predict(df_valid)
    
    h2o.export_file(y_train_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_train_local_pred.csv', force=True)
    h2o.export_file(y_test_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_test_local_pred.csv', force=True)
    h2o.export_file(y_valid_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_valid_local_pred.csv', force=True)
    
    # Default values
    if options.get("wanted") == None:
        wanted = None
    else:
        wanted = options.get("wanted")
        df_wanted = h2o.H2OFrame.from_python(wanted.tolist())#,column_names=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
        y_wanted_pred = dl_estimator.predict(df_wanted)
        h2o.export_file(y_wanted_pred, os.path.dirname(os.path.abspath(__file__))+'/temp/y_wanted_local_pred.csv', force=True)
        
def h2o_dl_readbw(weight_paths,bias_paths):
    
    for i in range(0,len(weight_paths)):
        if i == 0:
            weights = tuple(np.genfromtxt(weight_paths[i],delimiter=',',skip_header=1))
            biases = tuple(np.genfromtxt(bias_paths[i],delimiter=',',skip_header=1))
        else:
            weight = tuple(np.genfromtxt(weight_paths[i],delimiter=',',skip_header=1))
            bias = tuple(np.genfromtxt(bias_paths[i],delimiter=',',skip_header=1))
            biases = biases + bias
    return weights, biases
        

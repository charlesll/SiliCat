#! /usr/bin/env python
# Copyrights Le Losq, Di Genova 2016
# -*- coding: utf-8 -*-
import sys
import numpy as np

from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt

import silicat


def main():
    
    #%%
    # GET THE INPUT ARGUMENTS
    
    mode = sys.argv[1]
    ml_algo = sys.argv[2]
    
    print('\nRunning SiliCat Machine Learning Models with')
    print(mode+' mode')
    print(ml_algo+' algorithm')
    sys.stdout.flush()

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

    #%% MODEL STUFFS
    try:
        if mode == "train":
            if (ml_algo == "neurons"):
                silicat.ml.train_nn(X_input_train,Y_input_train,X_input_test,Y_input_test,'nn_global')
            elif (ml_algo == "svm"):
                silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'svm_global')
            elif (ml_algo == "gbr"):
                silicat.ml.train_gbr(X_input_train,Y_input_train,X_input_test,Y_input_test,'gbr_global')
            elif (ml_algo == "all"):
                silicat.ml.train_nn(X_input_train,Y_input_train,X_input_test,Y_input_test,'nn_global')
                silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'svm_global')
                silicat.ml.train_gbr(X_input_train,Y_input_train,X_input_test,Y_input_test,'gbr_global')
            else:
                raise NameError('NameError: second argument should be set to "neurons", "svm", "gbr" or "all"')
                
        elif mode == "predict":
            if (ml_algo == "neurons"):
                nn_model = silicat.ml.load_model('nn_global',origin="Keras")
            elif (ml_algo == "svm"):
                svm_model = silicat.ml.load_model('svm_global',origin="Scikit")
            elif (ml_algo == "gbr"):
                gbr_model = silicat.ml.load_model('gbr_global',origin="Scikit")
            elif (ml_algo == "all"):
                nn_model = silicat.ml.load_model('nn_global',origin="Keras")
                svm_model = silicat.ml.load_model('svm_global',origin="Scikit")
                gbr_model = silicat.ml.load_model('gbr_global',origin="Scikit")
            else:
                raise NameError('NameError: second argument should be set to "neurons", "svm", "gbr" or "all"')
                
        elif mode == "all":
            if (ml_algo == "neurons"):
                silicat.ml.train_nn(X_input_train,Y_input_train,X_input_test,Y_input_test,'nn_global')
                nn_model = silicat.ml.load_model('nn_global',origin="Keras")
            elif (ml_algo == "svm"):
                silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'svm_global') 
                svm_model = silicat.ml.load_model('svm_global',origin="Scikit")
            elif (ml_algo == "gbr"):
                silicat.ml.train_gbr(X_input_train,Y_input_train,X_input_test,Y_input_test,'gbr_global')     
                gbr_model = silicat.ml.load_model('gbr_global',origin="Scikit") 
            elif (ml_algo == "all"):
                silicat.ml.train_nn(X_input_train,Y_input_train,X_input_test,Y_input_test,'nn_global')
                nn_model = silicat.ml.load_model('nn_global',origin="Keras")
                silicat.ml.train_svm(X_input_train,Y_input_train,X_input_test,Y_input_test,'svm_global') 
                svm_model = silicat.ml.load_model('svm_global',origin="Scikit")
                silicat.ml.train_gbr(X_input_train,Y_input_train,X_input_test,Y_input_test,'gbr_global')     
                gbr_model = silicat.ml.load_model('gbr_global',origin="Scikit") 
            else:
                raise NameError('NameError: second argument should be set to "neurons", "svm", "gbr" or "all"')
                
        else:
            raise NameError('NameError: first argument should be set to "train", "predict" or "all"')
            
    except NameError as err:
        print(err.args) 

#%% MODEL OUTPUTS          
    if (mode == "predict" or mode == "all"):
        if (ml_algo == "neurons"):
            yp_train_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_train))
            yp_test_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_test))
            yp_valid_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_valid))
            yp_wanted_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_wanted))
            
            MSE_train_nn = np.sqrt(1./float(len(yp_train_nn)-1)*np.sum((yp_train_nn[:]-y_i_g[:])**2))
            MSE_test_nn = np.sqrt(1./float(len(yp_test_nn)-1)*np.sum((yp_test_nn[:]-y_t_g[:])**2))
            MSE_valid_nn = np.sqrt(1./float(len(yp_valid_nn)-1)*np.sum((yp_valid_nn[:]-y_v_g[:])**2))
            
            print('The training, testing and validating errors for the Neural Network model are: '+str(round(MSE_train_nn,2))+', '+str(round(MSE_test_nn,2))+', '+str(round(MSE_valid_nn,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_nn),(y_t_g,yp_test_nn),(y_v_g,yp_valid_nn),(user_wants[:,14],yp_wanted_nn),plot_title="Neural Network model")

        elif (ml_algo == "svm"):
            yp_train_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_train))
            yp_test_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_test))
            yp_valid_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_valid))
            yp_wanted_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_wanted))
            
            MSE_train_svm = np.sqrt(1./float(len(yp_train_svm)-1)*np.sum((yp_train_svm[:]-y_i_g[:,0])**2))
            MSE_test_svm = np.sqrt(1./float(len(yp_test_svm)-1)*np.sum((yp_test_svm[:]-y_t_g[:,0])**2))
            MSE_valid_svm = np.sqrt(1./float(len(yp_valid_svm)-1)*np.sum((yp_valid_svm[:]-y_v_g[:,0])**2))
            
            #print('Best parameters for svm model:\n')
            #print(svm_model.best_params_)
            #print()            
            
            print('The training, testing and validating errors for the SVM model are: '+str(round(MSE_train_svm,2))+', '+str(round(MSE_test_svm,2))+', '+str(round(MSE_valid_svm,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_svm),(y_t_g,yp_test_svm),(y_v_g,yp_valid_svm),(user_wants[:,14],yp_wanted_svm),plot_title="SVM model")

        elif (ml_algo == "gbr"):
            yp_train_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_train))
            yp_test_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_test))
            yp_valid_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_valid))
            yp_wanted_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_wanted))
            
            MSE_train_gbr = np.sqrt(1./float(len(yp_train_gbr)-1)*np.sum((yp_train_gbr[:]-y_i_g[:,0])**2))
            MSE_test_gbr = np.sqrt(1./float(len(yp_test_gbr)-1)*np.sum((yp_test_gbr[:]-y_t_g[:,0])**2))
            MSE_valid_gbr = np.sqrt(1./float(len(yp_valid_gbr)-1)*np.sum((yp_valid_gbr[:]-y_v_g[:,0])**2))
            
            #print('Best parameters for gbr model:\n')
            #print(gbr_model.best_params_)
            #print()                
            
            print('The training, testing and validating errors for the GBR model are: '+str(round(MSE_train_gbr,2))+', '+str(round(MSE_test_gbr,2))+', '+str(round(MSE_valid_gbr,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_gbr),(y_t_g,yp_test_gbr),(y_v_g,yp_valid_gbr),(user_wants[:,14],yp_wanted_gbr),plot_title="GBR model")
            
        elif (ml_algo == "all"):
            yp_train_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_train))
            yp_test_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_test))
            yp_valid_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_valid))
            yp_wanted_nn = scaler_y_g.inverse_transform(nn_model.predict(X_input_wanted))
            
            MSE_train_nn = np.sqrt(1./float(len(yp_train_nn)-1)*np.sum((yp_train_nn[:]-y_i_g[:])**2))
            MSE_test_nn = np.sqrt(1./float(len(yp_test_nn)-1)*np.sum((yp_test_nn[:]-y_t_g[:])**2))
            MSE_valid_nn = np.sqrt(1./float(len(yp_valid_nn)-1)*np.sum((yp_valid_nn[:]-y_v_g[:])**2))
            
            print('The training, testing and validating errors for the Neural Network model are: '+str(round(MSE_train_nn,2))+', '+str(round(MSE_test_nn,2))+', '+str(round(MSE_valid_nn,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_nn),(y_t_g,yp_test_nn),(y_v_g,yp_valid_nn),(user_wants[:,14],yp_wanted_nn),plot_title="Neural Network model")

            yp_train_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_train))
            yp_test_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_test))
            yp_valid_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_valid))
            yp_wanted_svm = scaler_y_g.inverse_transform(svm_model.predict(X_input_wanted))
            
            MSE_train_svm = np.sqrt(1./float(len(yp_train_svm)-1)*np.sum((yp_train_svm[:]-y_i_g[:,0])**2))
            MSE_test_svm = np.sqrt(1./float(len(yp_test_svm)-1)*np.sum((yp_test_svm[:]-y_t_g[:,0])**2))
            MSE_valid_svm = np.sqrt(1./float(len(yp_valid_svm)-1)*np.sum((yp_valid_svm[:]-y_v_g[:,0])**2))
            
#            print('Best parameters for svm model:\n')
#            print(svm_model.best_params_)
#            print()              
            
            print('The training, testing and validating errors for the SVM model are: '+str(round(MSE_train_svm,2))+', '+str(round(MSE_test_svm,2))+', '+str(round(MSE_valid_svm,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_svm),(y_t_g,yp_test_svm),(y_v_g,yp_valid_svm),(user_wants[:,14],yp_wanted_svm),plot_title="SVM model")

            yp_train_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_train))
            yp_test_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_test))
            yp_valid_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_valid))
            yp_wanted_gbr = scaler_y_g.inverse_transform(gbr_model.predict(X_input_wanted))
            
            MSE_train_gbr = np.sqrt(1./float(len(yp_train_gbr)-1)*np.sum((yp_train_gbr[:]-y_i_g[:,0])**2))
            MSE_test_gbr = np.sqrt(1./float(len(yp_test_gbr)-1)*np.sum((yp_test_gbr[:]-y_t_g[:,0])**2))
            MSE_valid_gbr = np.sqrt(1./float(len(yp_valid_gbr)-1)*np.sum((yp_valid_gbr[:]-y_v_g[:,0])**2))
            
#            print('Best parameters for gbr model:\n')
#            print(gbr_model.best_params_)
#            print()            
            
            print('The training, testing and validating errors for the GBR model are: '+str(round(MSE_train_gbr,2))+', '+str(round(MSE_test_gbr,2))+', '+str(round(MSE_valid_gbr,2))+'\n')
            silicat.ml.plot_model((y_i_g,yp_train_gbr),(y_t_g,yp_test_gbr),(y_v_g,yp_valid_gbr),(user_wants[:,14],yp_wanted_gbr),plot_title="GBR model")
            

if __name__ == "__main__":
   main()


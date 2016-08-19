# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:06:32 2016

@author: Charles Le Losq
"""

import pandas as pd
import sqlite3
import numpy as np

from sklearn import cross_validation, preprocessing
from sklearn.utils import shuffle

def chemical_splitting(Pandas_DataFrame, split_fraction):
    """
        This is a function to split dataset depending on their chemistry, to avoid the same chemical dataset to be
        found in different training/testing/validating datasets that are used in ML.
        
        Indeed, it is worthless to put data from the same original dataset / with the same chemical composition
        in the training / testing / validating datasets. This creates a initial bias in the splitting process...
         
        INPUTS:
        
        Pandas_DataFrame: A Pandas DataFrame 
            The input DataFrame with in the first row the names of the different data compositions
            
        split_fraction: a float number between 0 and 1
            This is the amount of splitting you want, in reference to the second output dataset (see OUTPUTS).            
            
        OUTPUTS:
            frame1 : A Pandas DataFrame 
                A DataSet with (1-split_fraction) datas from the input dataset with unique chemical composition / names
            
            frame2 : A Pandas DataFrame 
                A DataSet with split_fraction datas from the input dataset with unique chemical composition / names
                
            frame1_idx : A numpy array containing the indexes of the data picked in Pandas_DataFrame to construct frame1
            
            frame2_idx : A numpy array containing the indexes of the data picked in Pandas_DataFrame to construct frame2
    """
    names = Pandas_DataFrame['Name'].unique()
    names_idx = np.arange(len(names))

    # getting index for the frames with the help of scikitlearn
    frame1_idx, frame2_idx = cross_validation.train_test_split(names_idx, test_size = split_fraction)        
    
    # and now grabbing the relevant pandas dataframes
    ttt = np.in1d(Pandas_DataFrame.Name,names[frame1_idx])
    frame1 = Pandas_DataFrame[ttt == True]    
    
    ttt2 = np.in1d(Pandas_DataFrame.Name,names[frame2_idx])
    frame2 = Pandas_DataFrame[ttt2 == True] 
    
    return frame1, frame2, frame1_idx, frame2_idx  

def general_input(path):

    """
        This is a function to automate the reading of the viscosity database
         
        INPUTS:
        
        path: A string
            Path to the database
            
        OUTPUTS:
            scaler_x : scikitlearn scaler object
                Scaler for the X axis
            
            scaler_y : scikitlearn scaler object
                Scaler for the X axis
            
            X_train : Numpy array
                 X training data
            
            y_train : Numpy array
                Y training data
            
            X_test : Numpy array
                X testing data
            
            y_test : Numpy array
                Y testing data
            
            X_valid : Numpy array
                X validation data
            
            y_valid : Numpy array
                Y validation data
    """

    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(path)
    
    df = pd.read_sql_query("SELECT * from viscosity", con)
    
    # we first split the dataset in train, test and validation sub-datasets
    df_train, df_testvalid, train_idx, testvalid_idx = chemical_splitting(df, split_fraction = 0.3)
    df_test, df_valid, test_idx, valid_idx = chemical_splitting(df_testvalid, split_fraction = 0.5)
    
    con.close() # closing the sql connection

    data_train = pd.DataFrame(df_train,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    data_test = pd.DataFrame(df_test,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    data_valid = pd.DataFrame(df_valid,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])

    shuffle_train = shuffle(data_train, random_state=0)
    shuffle_test = shuffle(data_test, random_state=0)
    shuffle_valid = shuffle(data_valid, random_state=0)

    x_train = pd.DataFrame(shuffle_train,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_train = pd.DataFrame(shuffle_train,columns=['viscosity'])

    x_test = pd.DataFrame(shuffle_test,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_test = pd.DataFrame(shuffle_test,columns=['viscosity'])

    x_valid = pd.DataFrame(shuffle_valid,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_valid = pd.DataFrame(shuffle_valid,columns=['viscosity'])

    # scaling
    scaler_x = preprocessing.StandardScaler().fit(x_train.as_matrix())
    scaler_y = preprocessing.StandardScaler().fit(y_train.as_matrix()) 
    
    return scaler_x,scaler_y, x_train.as_matrix(), y_train.as_matrix(), x_test.as_matrix(), y_test.as_matrix(), x_valid.as_matrix(), y_valid.as_matrix()

def local_input(path, data_roi):

    """
        This is a function to automate the reading of the viscosity database
         
        INPUTS:
        
        path: A string
            Path to the database
            
        data_roi : Numpy array, 14x2
            the region of interest (roi) you want the data to be sampled in the global database; first column is the roi centre, second is the roi width
            
        OUTPUTS:
            scaler_x : scikitlearn scaler object
                Scaler for the X axis
            
            scaler_y : scikitlearn scaler object
                Scaler for the X axis
            
            X_train : Numpy array
                 X training data
            
            y_train : Numpy array
                Y training data
            
            X_test : Numpy array
                X testing data
            
            y_test : Numpy array
                Y testing data
            
            X_valid : Numpy array
                X validation data
            
            y_valid : Numpy array
                Y validation data
    """

    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(path)

    selection = (data_roi[0,0]-data_roi[0,1],data_roi[0,0]+data_roi[0,1],
              data_roi[1,0]-data_roi[1,1],data_roi[1,0]+data_roi[1,1],
              data_roi[2,0]-data_roi[2,1],data_roi[2,0]+data_roi[2,1],
              data_roi[3,0]-data_roi[3,1],data_roi[3,0]+data_roi[3,1],
              data_roi[4,0]-data_roi[4,1],data_roi[4,0]+data_roi[4,1],
              data_roi[5,0]-data_roi[5,1],data_roi[5,0]+data_roi[5,1],
              data_roi[6,0]-data_roi[6,1],data_roi[6,0]+data_roi[6,1],
              data_roi[7,0]-data_roi[7,1],data_roi[7,0]+data_roi[7,1],
              data_roi[8,0]-data_roi[8,1],data_roi[8,0]+data_roi[8,1],
              data_roi[9,0]-data_roi[9,1],data_roi[9,0]+data_roi[9,1],
              data_roi[10,0]-data_roi[10,1],data_roi[10,0]+data_roi[10,1],
              data_roi[11,0]-data_roi[11,1],data_roi[11,0]+data_roi[11,1],
              data_roi[12,0]-data_roi[12,1],data_roi[12,0]+data_roi[12,1],
              data_roi[13,0]-data_roi[13,1],data_roi[13,0]+data_roi[13,1])

    #df = pd.read_sql_query("SELECT * from visco_data WHERE %f<sio2<%f AND %f<tio2<%f AND %f<al2o3<%f AND %f<feot<%f AND %f<mno<%f AND %f<bao<%f AND %f<sro<%f AND %f<mgo<%f AND %f<cao<%f AND %f<li2o<%f AND %f<na2o<%f AND %f<k2o<%f AND %f<p2o5<%f AND %f<h2o<%f" %selection, con)
    df = pd.read_sql_query("SELECT * from viscosity WHERE %f<sio2 AND sio2<%f AND %f<tio2 AND tio2<%f AND %f<al2o3 AND al2o3<%f AND %f<feot AND feot<%f AND %f<mno AND mno<%f AND %f<bao AND bao<%f AND %f<sro AND sro<%f AND %f<mgo AND mgo<%f AND %f<cao AND cao<%f AND %f<li2o AND li2o<%f AND %f<na2o AND na2o<%f AND %f<k2o AND k2o<%f AND %f<p2o5 AND p2o5<%f AND %f<h2o AND h2o<%f" %selection, con)
    
    # we first split the dataset in train, test and validation sub-datasets
    df_train, df_testvalid, train_idx, testvalid_idx = chemical_splitting(df, split_fraction = 0.3)
    df_test, df_valid, test_idx, valid_idx = chemical_splitting(df_testvalid, split_fraction = 0.5)
    
    con.close() # closing the sql connection

    data_train = pd.DataFrame(df_train,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    data_test = pd.DataFrame(df_test,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])
    data_valid = pd.DataFrame(df_valid,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])

    shuffle_train = shuffle(data_train, random_state=0)
    shuffle_test = shuffle(data_test, random_state=0)
    shuffle_valid = shuffle(data_valid, random_state=0)

    x_train = pd.DataFrame(shuffle_train,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_train = pd.DataFrame(shuffle_train,columns=['viscosity'])

    x_test = pd.DataFrame(shuffle_test,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_test = pd.DataFrame(shuffle_test,columns=['viscosity'])

    x_valid = pd.DataFrame(shuffle_valid,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_valid = pd.DataFrame(shuffle_valid,columns=['viscosity'])

    # scaling
    scaler_x = preprocessing.StandardScaler().fit(x_train.as_matrix())
    scaler_y = preprocessing.StandardScaler().fit(y_train.as_matrix())    
    
    return scaler_x,scaler_y, x_train.as_matrix(), y_train.as_matrix(), x_test.as_matrix(), y_test.as_matrix(), x_valid.as_matrix(), y_valid.as_matrix()

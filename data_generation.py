# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:12:21 2016

@author: Charles Le Losq
"""
import sys
import numpy as np
from sklearn.externals import joblib
from silicat.data_import import general_input,local_input

#%%
###############################################################################
# Generate sample data
print("Generating the data files from the viscosity database, saving data subsets in ./data/viscosity/saved_subsets/ as pickle files")
sys.stdout.flush()

user_wants = np.genfromtxt("./inputs/wanted.csv",delimiter=',',skip_header=1)

selection = np.array([[user_wants[0,0],10.], 
                     [user_wants[0,1],10.],
                     [user_wants[0,2],10.],
                        [user_wants[0,3],10.],
                        [user_wants[0,4],10.],
                        [user_wants[0,5],10.],
                        [user_wants[0,6],10.],
                        [user_wants[0,7],10.],
                        [user_wants[0,8],10.],
                        [user_wants[0,9],10.],
                        [user_wants[0,10],10.],
                        [user_wants[0,11],10.],
                        [user_wants[0,12],10.],
                        [user_wants[0,13],10.]])
            
scaler_x_g, scaler_y_g, x_i_g, y_i_g, x_t_g, y_t_g, x_v_g, y_v_g = general_input("./data/viscosity/viscosity.sqlite")  
scaler_x_l, scaler_y_l, x_i_l, y_i_l, x_t_l, y_t_l, x_v_l, y_v_l = local_input("./data/viscosity/viscosity.sqlite",selection)

# for h2o
train_g  = np.concatenate((x_i_g,y_i_g),axis=1)
test_g  = np.concatenate((x_t_g,y_t_g),axis=1)
valid_g  = np.concatenate((x_v_g,y_v_g),axis=1)

train_l  = np.concatenate((x_i_l,y_i_l),axis=1)
test_l  = np.concatenate((x_t_l,y_t_l),axis=1)
valid_l  = np.concatenate((x_v_l,y_v_l),axis=1)

joblib.dump((scaler_x_g, scaler_y_g, x_i_g, y_i_g, x_t_g, y_t_g, x_v_g, y_v_g), './data/viscosity/saved_subsets/general_datas_sk.pkl') 
joblib.dump((scaler_x_l, scaler_y_l, x_i_l, y_i_l, x_t_l, y_t_l, x_v_l, y_v_l), './data/viscosity/saved_subsets/local_datas_sk.pkl') 

joblib.dump((train_g,test_g,valid_g), './data/viscosity/saved_subsets/general_datas_h2o.pkl') 
joblib.dump((train_l,test_l,valid_l), './data/viscosity/saved_subsets/local_datas_h2o.pkl') 

print("\nViscosity data subsets saved!")
sys.stdout.flush()

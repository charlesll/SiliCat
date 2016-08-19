#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:35:11 2016

@author: Charles Le Losq
"""
import sys
import numpy as np

from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import matplotlib
from matplotlib import pyplot as plt
gfont = {'fontname':'Geneva'}
afont = {'fontname':'Arial'}
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

import silicat

#%%
def main():    
    print('\nRunning SiliCat Data Visualisation script')
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

    #%% PLOTTING STUFFS
    fig = plt.figure(1,figsize=(12,12))
    plt.title('Current dataset')
    
    ax1 = plt.subplot(2,2,1)
    ax1.plot(10000/x_i_g[:,14],y_i_g[:,0],'b.', label='Training')
    ax1.plot(10000/x_t_g[:,14],y_t_g[:,0],'md',label='Testing')
    ax1.plot(10000/x_v_g[:,14],y_v_g[:,0],'cs', label='Validation')
    ax1.set_xlabel('10$^{4}$/T, K$^{-1}$',fontsize = 14, fontweight = 'bold',**afont)
    ax1.set_ylabel('Viscosity, log Pa s',fontsize = 14, fontweight = 'bold',**afont)
    ax1.set_title('Dataset',fontsize = 14, fontweight = 'bold',**afont)
    plt.legend(loc=4,frameon=False,fontsize=18)
    
    ax2 = plt.subplot(2,2,2)
    ax2.plot(X_input_train[:,0],X_input_train[:,10]+X_input_train[:,11],'b.', label='Training')
    ax2.plot(X_input_test[:,0],X_input_test[:,10]+X_input_test[:,11],'md', label='Testing')
    ax2.plot(X_input_valid[:,0],X_input_valid[:,10]+X_input_valid[:,11],'cs', label='Validation')
    ax2.set_xlabel('SiO$_2$, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax2.set_ylabel('K$_2$O+Na$_2$O, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax2.set_title('TAS diagram',fontsize = 14, fontweight = 'bold',**afont)
    
    ax3 = plt.subplot(2,2,3)
    ax3.plot(X_input_train[:,0]+X_input_train[:,1]+X_input_train[:,2],X_input_train[:,5]+X_input_train[:,6]+X_input_train[:,7]+X_input_train[:,8],'b.', label='Training')
    ax3.plot(X_input_test[:,0]+X_input_test[:,1]+X_input_test[:,2],X_input_test[:,5]+X_input_test[:,6]+X_input_test[:,7]+X_input_test[:,8],'md', label='Testing')    
    ax3.plot(X_input_valid[:,0]+X_input_valid[:,1]+X_input_valid[:,2],X_input_valid[:,5]+X_input_valid[:,6]+X_input_valid[:,7]+X_input_valid[:,8],'cs', label='Validation')    
    ax3.set_xlabel('SiO$_2$+Al$_2$O$_3$+TiO$_2$, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax3.set_ylabel('BaO+SrO+CaO+MgO, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax3.set_title('Network formers vs Alcaline-Earth elements',fontsize = 14, fontweight = 'bold',**afont)
    
    ax4 = plt.subplot(2,2,4)
    ax4.plot(X_input_train[:,0]+X_input_train[:,1]+X_input_train[:,2],X_input_train[:,9]+X_input_train[:,10]+X_input_train[:,11]+X_input_train[:,13],'b.', label='Training')
    ax4.plot(X_input_test[:,0]+X_input_test[:,1]+X_input_test[:,2],X_input_test[:,9]+X_input_test[:,10]+X_input_test[:,11]+X_input_test[:,13],'cd', label='Testing')    
    ax4.plot(X_input_valid[:,0]+X_input_valid[:,1]+X_input_valid[:,2],X_input_valid[:,9]+X_input_valid[:,10]+X_input_valid[:,11]+X_input_valid[:,13],'ms', label='Validation')    
    ax4.set_xlabel('SiO$_2$+Al$_2$O$_3$+TiO$_2$, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax4.set_ylabel('H$_2$O+Li$_2$O+Na$_2$O+K$_2$O, standardized',fontsize = 14, fontweight = 'bold',**afont)
    ax4.set_title('Network formers vs Alkali elements',fontsize = 14, fontweight = 'bold',**afont)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


#%%
if __name__ == "__main__":
   main()


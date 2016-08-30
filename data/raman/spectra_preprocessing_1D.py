# Copyrights Le Losq, Di Genova 2016
import numpy as np
import pickle as pkl

from scipy import interpolate
from scipy.interpolate import interp1d

import matplotlib
from matplotlib import pyplot as plt

#%%
# the next definition are important as they define the spectral region we look at and record in the global array
x_lf = np.arange(250.0,1350.0,2.0) # this is the low frequency of the common x axis
x_hf = np.arange(3000.0,3900.0,2.0) # this is the high frequency of the common x axis
x = np.concatenate((x_lf,x_hf),0) # this is the final common x axis


#%% 1D treatment
spectra_liste_sup = np.genfromtxt("spectra_labels.csv",delimiter=',',skip_header=1,dtype = 'string') # reading the list
spectra_1D_sup = np.zeros((len(spectra_liste_sup),len(x))) 
for i in range(0,len(spectra_liste_sup)): # starting the loop
    spectrum = np.genfromtxt("/Users/charles/OneDrive - Australian National University/spectra/"+spectra_liste_sup[i,0]) # we let genfromtxt guess the delimiter, hopping it works
    
    # resampling
    tck = interpolate.splrep(spectrum[:,0],spectrum[:,1],s=0) 
    signal = interpolate.splev(x,tck,der=0) # re-sampled signal with the good x axis        
    signal_corrected = signal - np.min(signal)
    spectra_1D_sup[i,:] =  signal_corrected/np.max(signal_corrected) # maximum is set to 1
    
f=open("spectra_1d_supervised",'w')
pkl.dump(spectra_1D_sup, f)
f.close()  

spectra_liste_unsup = np.genfromtxt("content.csv",delimiter=',',skip_header=1,dtype = 'string') # reading the list
spectra_1D_unsup = np.zeros((len(spectra_liste_unsup),len(x))) 
for i in range(0,len(spectra_liste_sup)): # starting the loop
    spectrum = np.genfromtxt("/Users/charles/OneDrive - Australian National University/spectra/"+spectra_liste_unsup[i]) # we let genfromtxt guess the delimiter, hopping it works
    
    # resampling
    tck = interpolate.splrep(spectrum[:,0],spectrum[:,1],s=0) 
    signal = interpolate.splev(x,tck,der=0) # re-sampled signal with the good x axis        
    signal_corrected = signal - np.min(signal)
    spectra_1D_unsup[i,:] =  signal_corrected/np.max(signal_corrected) # maximum is set to 1
    
f=open("spectra_1d_unsupervised",'w')
pkl.dump(spectra_1D_unsup, f)
f.close()  
# Copyrights Le Losq, Di Genova 2016
import numpy as np
import pickle as pkl
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

from scipy import interpolate
from scipy.interpolate import interp1d

import matplotlib
from matplotlib import pyplot as plt

#%% FUNCTION DEFINITION
##############################################################################
def longcorr(data,temp,wave): # input are a two column matrix of data, temperature in C, wave as the wavelength in cm-1
    """
    # temperature-wavelength Correction
    # Charles Le Losq
    # RSES Canberra 2016
    
    # See Shucker and Gammon, Phys. Rev. Lett. 1970; Galeener and Sen, Phys. Rev. B 1978; Neuville and Mysen, GCA 1996; Le Losq et al. AM 2012 and GCA 2014 for equations and theory
    """
    h = 6.62606896*10**-34   # J.s    Plank constant
    k = 1.38066e-23;     # J/K    Boltzman
    c = 2.9979e8;        # m/s    Speed of light
    v = wave;            # cm-1   Excitating laser line
    nu0 = 1.0/v*10**9;    # conversion of v in m
    T = temp + 273.15;   # C input temperature in K

    x = data[:,0]
    y = data[:,1]
       
    
    nu = 100.0*x; # cm-1 -> m-1 Raman shift
    rnu = nu0-nu; # nu0 is in m-1
    t0 = nu0*nu0*nu0*nu/rnu/rnu/rnu/rnu;
    t1 = -h*c*nu/k/T; # c in m/s  : t1 dimensionless
    t2 = 1 - np.exp(t1);
    longsp = y*t0*t2; # for y values
    
    return longsp
#%% 
##############################################################################
# names
type_data = "fe_sp"

# the next definition are important as they define the spectral region we look at and record in the global array
x_lf = np.arange(250.0,1300.0,2.0) # this is the low frequency of the common x axis
x_hf = np.arange(3000.0,3900.0,2.0) # this is the high frequency of the common x axis
x = np.concatenate((x_lf,x_hf),0) # this is the final common x axis

y_scale = np.arange(0,1,0.01) # Here a difference with a normal (regression) approach: we don't really care about the y axis values. So we are going to normalise to maximum intensity for convenience.

try:
    if type_data == "supervised":
        in_name = "spectra_labels.csv" # for supervised learning
        outname = "spectra_2d_supervised.pkl"
        spectra_liste = np.genfromtxt(in_name,delimiter=',',skip_header=1,dtype = 'string') # reading the list
        names = spectra_liste[:,0]
        spectra = np.zeros((len(spectra_liste),len(y_scale),len(x))) # this is the final array where spectra will be saved as a pickle file on the HDD.

    elif type_data == "unsupervised":
        in_name = "content.csv" # for pre-training
        outname = "spectra_2d_unsupervised.pkl"
        spectra_liste = np.genfromtxt(in_name,delimiter=',',skip_header=0,dtype = 'string') # reading the list
        names = spectra_liste[:]
        spectra = np.zeros((len(spectra_liste),len(y_scale),len(x))) # this is the final array where spectra will be saved as a pickle file on the HDD.

    elif type_data == "fe_sp":
        in_name = "fe_sp_std.csv" # for pre-training
        outname = "spectra_2d_sp.pkl"
        spectra_liste = np.genfromtxt(in_name,delimiter=',',skip_header=1,dtype = 'string') # reading the list
        names = spectra_liste[:,0]
        spectra = np.zeros((len(spectra_liste),len(y_scale),len(x_lf))) # this is the final array where spectra will be saved as a pickle file on the HDD.

    else:
        raise NameError('NameError: type_data should be set to "supervised", "unsupervised" or "fe_sp"')
        
except NameError as err:
        print(err.args) 

#%%
for i in range(0,len(spectra_liste)): # starting the loop

    spectrum = np.genfromtxt("/Users/charles/OneDrive - Australian National University/spectra/"+names[i]) # we let genfromtxt guess the delimiter, hopping it works
    #spectrum[:,1] = longcorr(spectrum,514.532,23.0)
    
    # checking that the signal is increasing, and correcting if necessary
    if spectrum[-1,0] < spectrum[0,0]:
        spectrum[:,:] = spectrum[::-1,:]
    
    # for test fe spectra
    if type_data == "fe_sp":
        # resampling
        tck = interpolate.splrep(spectrum[:,0],spectrum[:,1]) 
        signal = interpolate.splev(x_lf,tck,der=0) # re-sampled signal with the good x axis        
        signal_corrected = signal - np.min(signal)
    
        signal_corrected= signal_corrected
        signal_corrected = signal_corrected/np.max(signal_corrected) # maximum is set to 1
        # recording the spectra into the global Array
        for j in range(0,x_lf.size):
            spectra[i,np.where(y_scale<=signal_corrected[j]),j] = 1.0
        
    # for the other spectra with water peak    
    else:            
        #Checking if water signal is present, if not we put a false signal to 0
        test = spectrum[:,0] >= 3000.0
        if any(test) ==  False:
            water_signal = np.zeros((len(x_hf),2))
            water_signal[:,0]= x_hf[:]
            spectrum = np.concatenate((spectrum,water_signal),axis=0)
            
        # resampling
        tck = interpolate.splrep(spectrum[:,0],spectrum[:,1]) 
        signal = interpolate.splev(x,tck,der=0) # re-sampled signal with the good x axis        
        signal_corrected = signal - np.min(signal)
        
        signal_corrected= signal_corrected
        signal_corrected = signal_corrected/np.max(signal_corrected) # maximum is set to 1
        # recording the spectra into the global Array
        
        water_index = np.where(x==3000)    
        for j in range(0,water_index[0][0]):
            spectra[i,np.where(y_scale<=signal_corrected[j]),j] = 1.0
        for j in range(water_index[0][0],len(x)):
            spectra[i,np.where(y_scale<=signal_corrected[j]),j] = 0.5
#    for j in range(0,len(x)):
#        for k in range(0,len(y_scale)):
#            if np.isclose(y_scale[k],signal_corrected[j],rtol=0.02) == True:
#                spectra[i,k,j] = 1
                
f=open(outname,'w')
pkl.dump(spectra, f)
f.close()  
          
#%%
choice = 3 #spectrum number

plt.figure()
plt.imshow(spectra[choice,:,:])
plt.gray()



    #plt.figure()
    #plt.imshow(spectra[i,:,:])                
    #plt.ioff()
    #plt.figure()
    #plt.subplot(2,1,1)
    #plt.plot(x,signal,'k-')
    #plt.plot(x,baseline,'k-')
    #plt.subplot(2,1,2)
    #plt.plot(spectra[i,:,:],signal_corrected,'r-')
    #plt.xlabel("Raman schift, cm$^{-1}$")
    #plt.ylabel("Intensity")
    #plt.savefig("./figures/"+spectra_liste[i,0]+".jpg")
          
#          if switch == "complex":
#        # we perform a simple baseline subtraction, with using a constant for nu < 1300 cm-1, and a kernel ridge regression for the water peak
#        roi = np.array([[1200., 1350.],[2700., 3000.],[3750., 4000.]])
#        
#        const_portion = spectrum[np.where((spectrum[:,0]> roi[0,0]) & (spectrum[:,0] < roi[0,1]))]
#        lf_idx = np.argmin(const_portion[:,1])
#        lf_idx_xvalue = const_portion[lf_idx,0]
#        lf_const = np.mean(const_portion[lf_idx-10:lf_idx+10,1])
#        
#        water_portion_1 = spectrum[np.where((spectrum[:,0]> roi[1,0]) & (spectrum[:,0] < roi[1,1]))]
#        water_portion_2 = spectrum[np.where((spectrum[:,0]> roi[2,0]) & (spectrum[:,0] < roi[2,1]))]
#        water_portion = np.concatenate((water_portion_1,water_portion_2),axis=0)
#        
#        X_scaler = preprocessing.StandardScaler().fit(water_portion[:,0].reshape(-1,1))
#        y_scaler = preprocessing.StandardScaler().fit(water_portion[:,1].reshape(-1,1))
#    
#        X_i = X_scaler.transform(water_portion[:,0].reshape(-1,1))
#        y_i = y_scaler.transform(water_portion[:,1].reshape(-1,1))
#        
#        kr = GridSearchCV(KernelRidge(kernel='polynomial', gamma=0.1), cv=5,
#                          param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3, 1e-4],
#                                      "gamma": np.logspace(-2, 2, 5)})
#        
#        kr.fit(X_i, y_i[:].ravel())
#        
#        baseline_lf = np.ones(len(x_lf))*lf_const
#        baseline_hf = y_scaler.inverse_transform(kr.predict(X_scaler.transform(x_hf.reshape(-1,1))))
#        
#        baseline = np.concatenate((baseline_lf,baseline_hf),0)
#    
#        # resampling
#        tck = interpolate.splrep(spectrum[:,0],spectrum[:,1],s=0) 
#        signal = interpolate.splev(x,tck,der=0) # re-sampled signal with the good x axis
#        
#        # baseline correction
#        signal_corrected = signal - baseline 
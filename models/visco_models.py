# -*- coding: utf-8 -*-
"""
@author: Charles LE LOSQ
Created in 2015 at Carnegie Institution of Washington D.C.

Current affiliation: The Australian National University

Implementation of the Giordano et al. 2008 viscosity model in Python.

Reference:

Giordano, D., Russell, J. K., & Dingwell, D. B. (2008). Viscosity of magmatic liquids: a model. Earth and Planetary Science Letters, 271(1), 123-134.

"""

import numpy as np

def G2008(data): # for spectral fit
    """
    Compute the viscosity for a given chemical composition in mol% according to the Giordano et al. (2008) model
    
    Parameters
    ----------
    
    data : Numpy Array
        Arrays with in columns: mol% SiO2 TiO2 Al2O3 FeOt MnO MgO  CaO Na2O K2O P2O5 H2O
    
    Returns
    -------

    visco : Numpy Array
        The viscosity in log Pa s calculated with the Giordano et al. 2008 model
    Tg : double
        The glass transition temperature (viscosity = 10^12 Pa s)
    A : double
        Parameter A of the TVF equation
    B : double
        Parameter B of the TVF equation
    C : double
        Parameter C of the TVF equation
    
    Reference
    ---------
    Giordano, D., Russell, J. K., & Dingwell, D. B. (2008). Viscosity of magmatic liquids: a model. Earth and Planetary Science Letters, 271(1), 123-134.

    """
    #data as mol% SiO2 TiO2 Al2O3 FeOt MnO MgO  CaO Na2O K2O P2O5 H2O 
    # columns       0    1    2     3   4   5    6   7    8    9   10
    SiO2 = data[:,0]
    TiO2 = data[:,1]
    Al2O3 = data[:,2]
    FeOt = data[:,3]
    MnO = data[:,4]
    MgO = data[:,5]
    CaO = data[:,6]
    Na2O = data[:,7]
    K2O = data[:,8]
    P2O5 = data[:,9]
    H2O = data[:,10]    
    T = data[:,11]    
    
    #see table 1 of Giordano et al. for parameter significance
    b1 = 159.7
    b2 = -173.3
    b3 = 72.1
    b4 = 75.7
    b5 = -39    
    b6 = -84.1
    b7 = 141.5
    b11 = -2.43
    b12 = -0.91    
    b13 = 17.6
    c1 = 2.75
    c2 = 15.7
    c3 = 8.3
    c4 = 10.2
    c5 = -12.3
    c6 = -99.5
    c11 = 0.30
    
    V = H2O #we do not input F data such that it does not appear there.
    TA = TiO2 + Al2O3
    FM = FeOt + MnO + MgO
    NK = Na2O + K2O   
    
    bb1 = b1 * (SiO2+TiO2)
    bb2 = b2 * Al2O3
    bb3 = b3* (FeOt + MnO + P2O5)
    bb4 = MgO * b4
    bb5 = CaO * b5
    bb6 = (Na2O + V) * b6
    bb7 = (V+np.log(1+H2O))*b7
    bb11 = (SiO2+TiO2)*FM*b11
    bb12 = (SiO2+TA+P2O5)*(NK+H2O)*b12
    bb13 = (Al2O3*NK)*b13

    B = bb1 + bb2 + bb3 + bb4 + bb5 + bb6 + bb7 + bb11 + bb12 + bb13

    cc1 = SiO2 * c1
    cc2 = TA*c2
    cc3 = FM*c3
    cc4 = CaO *c4
    cc5 = NK * c5
    cc6 = np.log(1+V) * c6
    cc11 = (Al2O3 + FM + CaO - P2O5) * (NK + V) * c11
    
    C = cc1 + cc2 + cc3 + cc4 + cc5 + cc6 + cc11
    
    A = -4.55
    
    visco = A + B/(T-C)

    Tg = B/(12-A) + C    
    
    return visco, Tg, A, B, C




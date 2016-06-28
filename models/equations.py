# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:49:30 2016

@author: Charles Le Losq
"""

import numpy as np
from scipy.optimize import curve_fit

def tvf(t,A,B,T1): # TVF equation
    """
    The Tamman-Vogel-Fulcher equation log(n) = A + B/(t-T1)
    
    Parameters
    ----------
    A : double
    B : double
    T : Numpy array
        Temperatures in K
    T1 : double
    
    Returns
    -------
    log(n) : Numpy array
        Calculated viscosity, log Pa s
    """
    return A + B/(t-T1)
    
def tvf_fit(t,visco,t_fit, p0 = [-4,8000,500]): # fit the tvf equation to data
    """
    The fit of the Tamman-Vogel-Fulcher equation log(n) = A + B/(T-T1) to data.
    
    Parameters
    ----------
    t : Numpy array 
        Data temperature, K 
    visco : Numpy array D
        Data viscosity, log Pa s
    t_fit : Numpy array 
        Desirated temperature, K
    
    Options
    -------
    
    p0 : Array of initial values for [A, B, T1]. Defaults = [-4,8000,500]    
    
    Returns
    -------
    log(n) : Numpy Array 
        Calculated viscosity, log Pa s
    """
    popt, pcov = curve_fit(tvf,t,visco)
    return tvf(t_fit,popt[0],popt[1],popt[2])
    
def ag(t, Ae, Be, SconfTg, ap, b, Tg): # Adam and Gibbs equation, see Richet (1984)
    """
    The Adam and Gibbs equation for viscous flow. See Richet (1984).
    
    Parameters
    ----------
    t : Numpy array 
        Temperatures, K
    Ae : double 
        The viscosity at infinite temperature, log Pa s
    Be : double 
        Constant proportional to the activation energy of viscous flow, J/K
    SconfTg : double
        The configurational entropy at the glass transition temperature Tg, J/(mol K)
    ap : double
        The difference between the liquid Cp at T and the glass Cp at Tg, linear part, J/(mol K)
    b : double
        The temperature dependent part of the liquid Cp, J/(mol K)
    Tg : double
        The glass transition temperature, K    
    
    Returns
    -------
    log(n) : Numpy Array 
        Calculated viscosity, log Pa s
    """ 
    return Ae + Be / (t * (SconfTg + (ap * (np.log(t)-np.log(Tg)) + b * (t-Tg))))
    

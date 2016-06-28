# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:59:53 2016

@author: charles
"""

def RB1985(data,Tg):
    """
    The liquid heat capacity model of Richet and Bottinga (1985).
    
    Parameters
    ----------
    data : Numpy array 
        The chemical composition organised as
    Tg : double
        The glass transition temperature, K    
    
    Returns
    -------
    log(n) : Numpy Array 
        Calculated viscosity, log Pa s
        
    Reference
    ---------
    Richet, Pascal, and Yan Bottinga. 1985. “Heat Capacity of Aluminum-Free Liquid Silicates.” Geochimica et Cosmochimica Acta 49 (2): 471–86. doi:10.1016/0016-7037(85)90039-0.

    """ 
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]

    #Calcul des coeffs du Cp; avec les valeurs de Richet, CG 62, 1987, 111-124 et Richet, GCA 49, 1985, 471
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);    
    aCpl = 81.37*SiO2/100 + 27.21*Al2O3/100 + 100.6*Na2O/100 + 50.13*K2O/100 + SiO2/100*(K2O/100*K2O/100)*151.7 + 86.05*CaO/100 + 85.78*MgO/100;
    bCpl = 0.0943*Al2O3/100 + 0.01578*K2O/100;
    ap = aCpl - Cpg;
    b = bCpl;

    return ap, b
    
def R1987(datachem,Tg):
    """
    The glass heat capacity model of Richet (1987).
    
    Parameters
    ----------
    data : Numpy array 
        The chemical composition organised as
    Tg : double
        The glass transition temperature, K    
    
    Returns
    -------
    log(n) : Numpy Array 
        Calculated viscosity, log Pa s
        
    Reference
    ---------
    Richet, Pascal. 1987. “Heat Capacity of Silicate Glasses.” Chemical Geology 62 (1): 111–24. doi:10.1016/0009-2541(87)90062-3.

    """
  
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]
    
    sio2_coefs = [127.200,-0.010777,431270.0,-1463.9]
    #n2o_coeffs = []
    #sio2_b = -0.010777

    #Calcul des coeffs du Cp; avec les valeurs de Richet, CG 62, 1987, 111-124 et Richet, GCA 49, 1985, 471
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);    
    aCpl = 81.37*SiO2/100 + 27.21*Al2O3/100 + 100.6*Na2O/100 + 50.13*K2O/100 + SiO2/100*(K2O/100*K2O/100)*151.7 + 86.05*CaO/100 + 85.78*MgO/100;
    bCpl = 0.0943*Al2O3/100 + 0.01578*K2O/100;
    ap = aCpl - Cpg;
    b = bCpl;

    return ap, b
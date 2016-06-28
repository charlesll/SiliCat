# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:27:35 2016

@author: Charles Le Losq
"""

import h2o
from data_import import general_input

# Start H2O on your local machine
h2o.init()

###############################################################################
# Generate sample data
x_i, y_i, x_t, y_t, x_v, y_v = general_input()

###############################################################################
# Generate sample data in H2O Frame
#h2o.H2OFrame(f.values.tolist())   # get no header
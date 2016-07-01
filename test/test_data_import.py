# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:33:29 2016

@author: charles
"""
import unittest, sys, os, sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from silicat.data_import import chemical_splitting

def fun_split_test1():
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("../data/viscosity/viscosity.sqlite")
    
    df = pd.read_sql_query("SELECT * from visco_data", con)
    
    # we first split the dataset in train, test and validation sub-datasets
    df_train, df_testvalid = chemical_splitting(df, split_fraction = 0.3)
    df_test, df_valid = chemical_splitting(df_testvalid, split_fraction = 0.5)

    con.close()
    return len(df['Name'].unique()), len(df_train['Name'].unique()) + len(df_test['Name'].unique()) + len(df_valid['Name'].unique())

class MyTest(unittest.TestCase):
    def test(self):
        len_tot, len_sum = fun_split_test1()
        self.assertEqual(len_tot, len_sum)
        
if __name__ == '__main__':
    unittest.main()
        
        

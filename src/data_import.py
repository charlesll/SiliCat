# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:06:32 2016

@author: Charles Le Losq
"""

import pandas as pd
import sqlite3

from sklearn import cross_validation
from sklearn.utils import shuffle

def general_input():

    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("../data/viscosity/viscosity.sqlite")

    df = pd.read_sql_query("SELECT * from visco_data", con)

    # verify that result of SQL query is stored in the dataframe
    print(df.head())

    con.close()

    data_total = pd.DataFrame(df,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T','viscosity'])

    shuffle_total = shuffle(data_total, random_state=0)

    x_shuffle_total = pd.DataFrame(shuffle_total,columns=['sio2','tio2','al2o3','feot','mno','bao','sro','mgo','cao','li2o','na2o','k2o','p2o5','h2o','T'])
    y_shuffle_total = pd.DataFrame(shuffle_total,columns=['viscosity'])

    X_train, X_2, y_train, y_2 = cross_validation.train_test_split(x_shuffle_total, y_shuffle_total, test_size=0.40, random_state=0)
    X_test, X_valid, y_test, y_valid = cross_validation.train_test_split(X_2, y_2, test_size=0.50, random_state=0)
    return X_train, y_train, X_test, y_test, X_valid, y_valid

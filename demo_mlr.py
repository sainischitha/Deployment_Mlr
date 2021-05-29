# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:25:17 2021

@author: ssridhar
"""


import pandas as pd
import numpy as np
from sklearn import linear_model
#import requests
import pickle

lm = linear_model.LinearRegression()

x1 = pd.Series([1,2,3,4,5])
x2 = pd.Series([2,3,5,8,11])
y = pd.Series([5,12,13,23,24])
X = pd.DataFrame({'x1':x1, 'x2':x2})
X.x2 = np.log(X.x2)
lm.fit(X,y)

lm.coef_, lm.intercept_

lm.predict(X)

# save the model to disk
filename = 'demo_mlr.pkl'
pickle.dump(lm, open(filename, 'wb')) 
    
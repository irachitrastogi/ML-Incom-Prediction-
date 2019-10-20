# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:57:56 2019

@author: irachitrastogi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import xgboost as xgb
import math

dataset = pd.read_csv('input_data.csv')
#data_raw = dataset.drop(['Profession','Size of City'],axis=1)
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, -1].values

z=X[0:111993,:]
z1=pd.DataFrame(z)
z1.tail()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='median') #control+i is used for inspection
imputer = imputer.fit(X[:, [0, 2]])
X[:, [0, 2]] = imputer.transform(X[:, [0, 2]])
df=pd.DataFrame(X)

X[pd.isnull(X)]  = 'NaN'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_5 = LabelEncoder()
X[:, 6] = labelencoder_X_5.fit_transform(X[:, 6])
#labelencoder_X_5 = LabelEncoder()
#X[:, 8] = labelencoder_X_5.fit_transform(X[:, 8])

split_data=X
X=X[0:111993,:]
y=y[0:111993]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


d_train = xgb.DMatrix(X_train, y_train)
d_test = xgb.DMatrix(X_test)

 params_1 = {"objective":"reg:linear",
             'colsample_bytree': 1,
             'learning_rate': 0.15,
             'booster': 'gbtree',
             'base_score':0.5,
             'alpha': 5, 
             'max_depth' : 25, 
             'min_child_weight': 7, 
             'n_estimators' : 1000, 
             'Gamma': 4, 
             'Subsample' :0.6,  
             'early_stopping_rounds':10}
 
 xgb_reg = xgb.train(params_1, d_train, num_boost_round=100)
 preds = xgb_reg.predict(d_test)
 

math.sqrt(((preds-y_test)**2).mean())

z=pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
k_new = pd.DataFrame(z)

dt3=split_data[111993:,:]
final_test = xgb.DMatrix(dt3)
y_pred_final = xgb_reg.predict(final_test)
Instance = k_new.Instance
ans = np.stack((Instance,y_pred_final),axis= 1)
ans = pd.DataFrame(ans)
ans.to_csv('Submit_XGboost2.csv')


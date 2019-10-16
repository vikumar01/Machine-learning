import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectKBest, f_regression

dataset=pd.read_csv('training_data.csv')
dataset
dataset.isnull().any()

from sklearn.impute import SimpleImputer
simpleimputermedian=SimpleImputer(strategy='mean')
dataset['Year of Record']=simpleimputermedian.fit_transform(dataset['Year of Record'].values.reshape(-1,1))
dataset['Age']=simpleimputermedian.fit_transform(dataset['Age'].values.reshape(-1,1))
dataset['Body Height [cm]']=simpleimputermedian.fit_transform(dataset['Body Height [cm]'].values.reshape(-1,1))
datasetnoncateg=dataset.drop(['Instance','Hair Color','Wears Glasses',],axis=1)


M=pd.read_csv('prediction_data.csv')
M['Year of Record']=simpleimputermedian.fit_transform(M['Year of Record'].values.reshape(-1,1))
M['Age']=simpleimputermedian.fit_transform(M['Age'].values.reshape(-1,1))
M['Body Height [cm]']=simpleimputermedian.fit_transform(M['Body Height [cm]'].values.reshape(-1,1))
Mnoncateg=M.drop(['Instance','Hair Color','Wears Glasses','Hair Color','Income'],axis=1)


X=datasetnoncateg.drop('Income in EUR',axis=1).values
Y=datasetnoncateg['Income in EUR'].values
#target encoding
t1 = TargetEncoder()
t1.fit(X, Y)
X = t1.transform(X)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=0)
# regressor = BayesianRidge()
regressor = RandomForestRegressor()
#regressor = AdaBoostRegressor()
#regressor = = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)

fitResult = regressor.fit(Xtrain, Ytrain)
YPredTest = regressor.predict(Xtest)
#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))


A=Mnoncateg.values
A=t1.transform(A)
B=regressor.predict(A)

df2=pd.DataFrame()
df2['Instance']=M['Instance']
df2['Income']=B

df2.to_csv(r'output1.csv',index=False)

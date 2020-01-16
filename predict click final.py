# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:39:04 2019

@author: Deepak Gupta
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#curent disc
os.chdir("E:\\predict click")
#import csv file
dataset_train=pd.read_csv("train_set.csv")
dataset_train.head()
dataset_train.info()
#to check null value by heat map
sns.heatmap(dataset_train.isnull(),yticklabels=False,cbar=False, cmap='coolwarm')
missing_value=dataset_train.isnull().sum()
missing_value = missing_value.reset_index()
missing_value = missing_value.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
#find out percentage of null value
missing_value['Missing_percentage'] = (missing_value['Missing_percentage']/len(dataset_train))*100
missing_value = missing_value.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
#drop unique id
dataset_train=dataset_train.drop("hotel_id",axis=1)
#fill null value 
dataset_train["stars"]=dataset_train["stars"].fillna(dataset_train["stars"].mode()[0])
dataset_train.fillna(value=dataset_train.median(),inplace=True)
# to check null value
dataset_train.isnull().sum()
#correlation
corr = dataset_train.corr()
#corrilation check by heatmap
f,ax=plt.subplots(figsize=(7,5))
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)    
plt.show()
#drop variable with two high indepandent varibale
columns=np.full((corr.shape[0],),True,dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1,corr.shape[0]):
        if corr.iloc[i,j]>=0.9:
            if columns [j]:
                columns[j]=False
                
selected_columns=dataset_train.columns[columns]
data_train=dataset_train[selected_columns]
#to check outliers
plt.boxplot(data_train["avg_saving_percent"])
  #outlier replace by lower boundry of upper boundry     
def outlier_defect(df):
    for i in df.describe().columns:
        q1=df.describe().at["25%",i]
        q3=df.describe().at["75%",i]
        IQR=(q3-q1)
        ltv=(q1-1.5*IQR)
        utv=(q3+1.5*IQR)
        x=np.array(df[i])
        p=[]
        for j in x:
             if j<ltv:
                p.append(ltv)
             elif j>utv:
                p.append(utv)
             else:
                p.append(j)
        df[i]=p
    return(df)

dataset_1=outlier_defect(data_train)
#check agin outliers
plt.boxplot(dataset_1["avg_saving_percent"])

#import library 
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#split dataset into x,y 
x=dataset_1.iloc[:,0:10]
y=dataset_1.iloc[:,10]
#train ,test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
#apply linaer regression 
ln=LinearRegression()
#fit the model
ln=ln.fit(x_train,y_train)
#to predict value 
y_pred=ln.predict(x_test)
#import library for calculated rmse
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
#cross validation for remove overfitting
from sklearn.model_selection import cross_val_score
scores=cross_val_score(ln,x_test,y_test,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def print_scores(scores):
    print("scores:", scores)
    print("mean:",scores.mean())
    print("std",scores.std())
    
print_scores(rmse_scores)    

#apply decision tree#********************************************************************
   
dtr=DecisionTreeRegressor()
model=dtr.fit(x_train,y_train)    
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
#cross validation for remove overfitting
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x_test,y_test,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def print_scores(scores):
    print("scores:", scores)
    print("mean:",scores.mean())
    print("std",scores.std())
    
print_scores(rmse_scores)    

# random forest#**************************************************************************8
rfc = RandomForestRegressor()
rfc_fit =rfc.fit(x_train,y_train)

y_pred=rfc_fit.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
#cross validation
scores=cross_val_score(rfc_fit,x_test,y_test,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def print_scores(scores):
    print("scores:", scores)
    print("mean:",scores.mean())
    print("std",scores.std())
    
print_scores(rmse_scores)   





# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:19:51 2020

@author: Deepak Gupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os
from sklearn import metrics
import seaborn as sns
from scipy.stats import chi2_contingency

os.getcwd()
os.chdir("E:\\PYTHON NOTES\\projects")
dataset=pd.read_csv("musk_csv.csv")
dataset.describe()
# dimension of data 
dataset.shape
# Number of rows
dataset.shape[0]
# number of columns
dataset.shape[1]
# name of columns
list(dataset)
# data detailat
dataset.info()

dataset.isnull().sum()
dataset=dataset.drop(["molecule_name","conformation_name"],axis=1 )
dataset1=dataset["class"] 
dataset=dataset.drop("class",axis=1 )
#outlier detection 
def outlier_detect(df):
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
    return (df) 

dataset_1=outlier_detect(dataset)
#find correlation 
corr=dataset.corr()
f,ax=plt.subplots(figsize=(7,5))             
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)

#check corrilation  and drop varibale which is high corrilated 
columns=np.full((corr.shape[0],),True,dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1,corr.shape[0]):
        if corr.iloc[i,j]>=0.9:
            if columns [j]:
                columns[j]=False
                
                
selected_columns=dataset.columns[columns]
final_df=dataset[selected_columns]

#apply decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x=final_df
y=dataset1
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2)
# apply decsion tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
C50_Predictions = C50_model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,C50_Predictions))
X=x.columns
dot_data = tree.export_graphviz(C50_model, out_file=None, max_depth=3, feature_names=X, class_names=['1','0'],filled=True, rounded=True,special_characters=True)
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
graph2 = graphviz.Source(dot_data)
graph2.render("final")

params = {'max_features': ['auto', 'sqrt', 'log2'],'min_samples_split': [2,3,4,5,6,7,8,9,10], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],'max_depth':[2,3,4,5,6,7,8,9]}

params


# Initializing Decision Tree
#gridsearch cv
DTC = tree.DecisionTreeClassifier()
# Building and Fitting Model

DTC1 = GridSearchCV(DTC, param_grid=params)
DTC_RS=DTC1.fit(x_train,y_train)

modelF = DTC_RS.best_estimator_
modelF

pred_modelF = modelF.predict(x_test)
metrics.accuracy_score(y_test,pred_modelF)          

from scipy.stats import randint as sp_randint
param_grid2 = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': sp_randint(2,10), 
          'min_samples_leaf': sp_randint(1,11),
         'max_depth':sp_randint(2,8)}

DTC_RS = RandomizedSearchCV(DTC, param_distributions=param_grid2,n_iter=100)
DTC_RS1 = DTC_RS.fit(x_train,y_train)
model2=DTC_RS1.best_estimator_
pred_model2=model2.predict(x_test)
metrics.accuracy_score(y_test,pred_model2)
# 0.9348484848484848
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,pred_modelF )
auc(false_positive_rate, true_positive_rate)
confusion_metrics=metrics.confusion_matrix(y_test,pred_modelF)
confusion_metrics

class_name=[0,1]
fig,ax=plt.subplots()
ticks_marks=np.arange(len(class_name))
plt.xticks(ticks_marks, class_name)
plt.yticks(ticks_marks, class_name)
sns.heatmap(pd.DataFrame(confusion_metrics),annot=True,cmap="YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y=1.1)
plt.ylabel("actual label")
plt.xlabel("predicted label")

# accuracy
print("Accuracy:",metrics.accuracy_score(y_test,pred_model2)) #0.9348484848484848
print("Precision:",metrics.precision_score(y_test,pred_model2)) #0.9078014184397163
print("Recall:",metrics.recall_score(y_test, pred_model2)) #0.6368159203980099
Accuracy=metrics.accuracy_score(y_test,pred_model2)
Precision=metrics.precision_score(y_test,pred_model2)
Recall=metrics.recall_score(y_test, pred_model2)
f1_score = 2*((Recall+Precision)/(Recall*Precision))
f1_score # 5.343750000000001
#apply random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=30)
rfc_fit=rfc.fit(x_train,y_train)
#prediction on train
rfc_pred_train =rfc_fit.predict(x_train)
# Accuracy on train
train_accuracy =metrics.accuracy_score(y_train,rfc_pred_train)
train_accuracy
# prediction on test
rfc_pred = rfc_fit.predict(x_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred)
test_accuracy
#using hyperparamter
#gridsearch and random searc
params_RF = {"max_depth": [3,5,6,7,8], "max_features":['auto', 'sqrt', 'log2'],
"min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10],
"criterion": ["gini", "entropy"]}
params_RF
model_RF = GridSearchCV(RandomForestClassifier(), param_grid=params_RF)
model_RF.fit(x_train,y_train)
# Best Parameters
	
model_RF.best_params_
# Predict and Check Accuracy for train
rfc_pred_train1 =model_RF.predict(x_train)
train_accuracy1 =metrics.accuracy_score(y_train,rfc_pred_train1)
train_accuracy
# prediction on test
rfc_pred1 = model_RF.predict(x_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred1)
test_accuracy

# Random Search
# Parameters
from scipy.stats import randint 
params_RF_RS = {"max_depth": randint(3,8),
"max_features":['auto', 'sqrt', 'log2'], "min_samples_split":randint (2,10),
"min_samples_leaf":randint (1,10),
"criterion": ["gini", "entropy"]}
# Building and Fitting Model
RF_RS = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params_RF_RS,n_iter=100)
RF_RS.fit(x_train,y_train)
# Best Parameters
RF_RS.best_params_
# Predict and Check Accuracy
# Predict and Check Accuracy for train
rfc_pred_train2 =RF_RS.predict(x_train)
train_accuracy2 =metrics.accuracy_score(y_train,rfc_pred_train2)
train_accuracy
# prediction on test
rfc_pred2 = RF_RS.predict(x_test)

# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred2)
test_accuracy
#0.9901515151515151 
print("Accuracy:",metrics.accuracy_score(y_test,rfc_pred2)) #0.9901515151515151
print("Precision:",metrics.precision_score(y_test,rfc_pred2)) #0.9895833333333334
print("Recall:",metrics.recall_score(y_test, rfc_pred2)) # 0.945273631840796
Accuracy=metrics.accuracy_score(y_test,rfc_pred2)
Precision=metrics.precision_score(y_test,rfc_pred2)
Recall=metrics.recall_score(y_test, rfc_pred2)
f1_score = 2*((Recall+Precision)/(Recall*Precision))
f1_score #  4.136842105263158

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:14:12 2019

@author: Deepak Gupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import os
import seaborn as sns
from scipy.stats import chi2_contingency
os.chdir("E:\PYTHON NOTES\projects\cab fare prediction")
dataset_train=pd.read_csv("train_cab.csv")
dataset_test=pd.read_csv("test.csv")

dataset_train.describe()
# dimension of data 
# dimension of data 
dataset_train.shape
# Number of rows
dataset_train.shape[0]
# number of columns
dataset_train.shape[1]
# name of columns
list(dataset_train)
# data detailat
dataset_train.info()


dataset_train.isnull().sum()
dataset_test.isnull().sum()
sns.heatmap(dataset_train.isnull(),yticklabels=False,cbar=False, cmap='coolwarm')

 #datetime change into reqired format
data=[dataset_train,dataset_test]
for i in data:
    i["pickup_datetime"]=pd.to_datetime(i["pickup_datetime"],errors="coerce")
 
    
dataset_train.info() 
dataset_test.info()  
dataset_train.isnull().sum()
dataset_test.isna().sum()
dataset_train=dataset_train.dropna(subset=["pickup_datetime"],how="all")
dataset_train["fare_amount"]=dataset_train["fare_amount"].astype(float)
np.where(dataset_train["fare_amount"]=="430-")
dataset_train["fare_amount"].loc[1123]=430
dataset_train["fare_amount"]=dataset_train["fare_amount"].astype(float)
#we will convery passanger count in to catogorical varibale ,cause passangor caount is not contineous varibale 
dataset_obj=["passenger_count"]
dataset_int=["fare_amount","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]
 # data visulization
import seaborn as sns
import matplotlib.pyplot as plt
#$stting up the sns for plots
sns.set(style="darkgrid",palette="Set1")

#some histogram plot from seaborn lib
plt.figure(figsize=(20,20))
plt.subplot(321)
_=sns.distplot(dataset_train["fare_amount"],bins=50)
plt.subplot(322)
_=sns.distplot(dataset_train["pickup_longitude"],bins=50)
plt.subplot(323)
_=sns.distplot(dataset_train["pickup_latitude"],bins=50)
plt.subplot(324)
_ = sns.distplot(dataset_train['dropoff_longitude'],bins=50)
plt.subplot(325)
_ = sns.distplot(dataset_train['dropoff_latitude'],bins=50)

plt.show()
plt.savefig('hist.png')

import scipy.stats as stats

#Some Bee Swarmplots
# plt.title('Cab Fare w.r.t passenger_count')

plt.figure(figsize=(25,25))
#_=sns.swarmplot(x="passenger_count",y="fare_amount",data=dataset_train)

#Jointplots for Bivariate Analysis.
#Here Scatter plot has regression line between 2 variables along with separate Bar plots of both variables.
#Also its annotated with pearson correlation coefficient and p value.

_=sns.jointplot(x="fare_amount",y="pickup_longitude",data=dataset_train,kind="reg")
_.annotate(stats.pearsonr)
#plt.savefig("jointfplo.png")
plt.show()
_=sns.jointplot(x="fare_amount",y="pickup_latitude",data=dataset_train,kind="reg")
_.annotate(stats.pearsonr)

_=sns.jointplot(x="fare_amount",y="dropoff_longitude",data=dataset_train,kind="reg")
_.annotate(stats.pearsonr)
_=sns.jointplot(x="fare_amount",y="dropoff_latitude",data=dataset_train,kind="reg")
_.annotate(stats.pearsonr)
 #  some violineplots to see spread d variable
plt.figure(figsize=(20,20))
plt.subplot(321)
_=sns.violinplot(y="fare_amount",data=dataset_train)
plt.subplot(322)
_=sns.violinplot(y="pickup_longitude",data=dataset_train)
plt.subplot(323)
_ = sns.violinplot(y='pickup_latitude',data=dataset_train)
plt.subplot(324)
_ = sns.violinplot(y='dropoff_longitude',data=dataset_train)
plt.subplot(325)
_ = sns.violinplot(y='dropoff_latitude',data=dataset_train)
plt.savefig("violine.png")
plt.show()

#pairplot for all numeric varibale
_=sns.pairplot(dataset_train.loc[:,dataset_int],kind="scatter",dropna=True)
_.fig.suptitle("pairwise plot all numeric varibale")
#plt.savefig("pairwise.png")
plt.show()
 #removing values which are not within the desired range outlier depanding upon basic understanding of dataset
#1.Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0. So we will remove these fields. 

sum(dataset_train["fare_amount"]<1)
dataset_train[dataset_train["fare_amount"]<1]
dataset_train=dataset_train.drop(dataset_train[dataset_train["fare_amount"]<1].index,axis=0)

#dataset_train.loc[dataset_train["fare_amount"]<1,"fare_amount"]=np.nan

#2. passanger count varibale /// passanger count cound not increse more than 6

sum(dataset_train["passenger_count"]>6)


for i in range (4,11):
     print("passanger_count_above"+ str(i)+ "={}".format(sum(dataset_train["passenger_count"]>i)))
     
#  so 20 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them. 

dataset_train[dataset_train["passenger_count"]>6]
#Also we need to see if there are any passenger_count<1
dataset_train[dataset_train["passenger_count"]<1]
len(dataset_train[dataset_train["passenger_count"]<1])
dataset_test["passenger_count"].unique()
# We will remove 20 observation which are above 6 value because a cab cannot hold these number of passengers.
dataset_train=dataset_train.drop(dataset_train[dataset_train["passenger_count"]<1].index,axis=0)
dataset_train=dataset_train.drop(dataset_train[dataset_train["passenger_count"]>6].index,axis=0)
#dataset_train.loc[dataset_train["passenger_count"]<1,"passenger_count"]=np.nan
#dataset_train.loc[dataset_train["passenger_count"]>6,"passenger_count"]=np.nan
sum(dataset_train["passenger_count"]<1)
#3.Latitudes range from -90 to 90.Longitudes range from -180 to 180. Removing which does not satisfy these ranges
print("pickup_longitude above 180  ={}".format(sum(dataset_train["pickup_longitude"]>180)))
print("pickup_longitude above -180 = {}".format(sum(dataset_train["pickup_longitude"]<-180)))
print("pickup_latitude above 90 ={}".format(sum(dataset_train["pickup_latitude"]>90)))
print("pickup_latitude above -90 ={}".format(sum(dataset_train["pickup_latitude"]<-90)))
print('dropoff_longitude above 180={}'.format(sum(dataset_train['dropoff_longitude']>180)))
print('dropoff_longitude below -180={}'.format(sum(dataset_train['dropoff_longitude']<-180)))
print('dropoff_latitude below -90={}'.format(sum(dataset_train['dropoff_latitude']<-90)))
print('dropoff_latitude above 90={}'.format(sum(dataset_train['dropoff_latitude']>90)))

#for test data


print("pickup_longitude above 180  ={}".format(sum(dataset_test["pickup_longitude"]>180)))
print("pickup_longitude above -180 = {}".format(sum(dataset_test["pickup_longitude"]<-180)))
print("pickup_latitude above 90 ={}".format(sum(dataset_test["pickup_latitude"]>90)))
print("pickup_latitude above -90 ={}".format(sum(dataset_test["pickup_latitude"]<-90)))
print('dropoff_longitude above 180={}'.format(sum(dataset_test['dropoff_longitude']>180)))
print('dropoff_longitude below -180={}'.format(sum(dataset_test['dropoff_longitude']<-180)))
print('dropoff_latitude below -90={}'.format(sum(dataset_test['dropoff_latitude']<-90)))
print('dropoff_latitude above 90={}'.format(sum(dataset_test['dropoff_latitude']>90)))







#There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.
#Also we will see if there are any values equal to 0.

for i in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
    print(i,"equal to 0={}".format(sum(dataset_train[i]==0)))


#for test data 
    
for i in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
    print(i,"equal to 0={}".format(sum(dataset_test[i]==0)))   
    
    
#there are values which are equal to 0. we will remove them.
# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan   
dataset_train=dataset_train.drop(dataset_train[dataset_train["pickup_latitude"]>90].index,axis=0)

#there are values which are equal to 0. we will remove them.
    
for i in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
    dataset_train=dataset_train.drop(dataset_train[dataset_train[i]==0].index,axis=0)

# for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
#     train.loc[train[i]==0,i] = np.nan
# train.loc[train['pickup_latitude']>90,'pickup_latitude'] = np.nan
    
dataset_train.shape    

#Missing Value Analysis    


missing_value=dataset_train.isnull().sum()
missing_value = missing_value.reset_index()
missing_value = missing_value.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
#find out percentage of null value
missing_value['Missing_percentage'] = (missing_value['Missing_percentage']/len(dataset_train))*100
missing_value = missing_value.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

dataset_train.info()
dataset_train["fare_amount"]=dataset_train["fare_amount"].fillna(dataset_train["fare_amount"].median())

dataset_train["passenger_count"]=dataset_train["passenger_count"].fillna(dataset_train["passenger_count"].mode()[0])

dataset_train.isnull().sum()
dataset_train["passenger_count"]=dataset_train["passenger_count"].round().astype(object)
dataset_train["passenger_count"].unique()

#outliers analysis by box plot
plt.figure(figsize=(20,5))
plt.xlim(0,100)
sns.boxplot(x=dataset_train["fare_amount"],data=dataset_train,orient="h")

# sum(dataset_train['fare_amount']<22.5)/len(dataset_train['fare_amount'])*100

#Bivariate Boxplots: Boxplot for Numerical Variable Vs Categorical Variable.
plt.figure(figsize=(20,10))
plt.xlim(0,100)
_=sns.boxplot(x=dataset_train["fare_amount"],y=dataset_train["passenger_count"],data=dataset_train,orient="h")


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


dataset_int1=outlier_detect(dataset_train.loc[:,dataset_int])

dataset_test_obj=["passenger_count"]
dataset_test_int=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]

dataset_test1=outlier_detect(dataset_test.loc[:,dataset_test_int])
dataset_test1=pd.concat([dataset_test1,dataset_test["passenger_count"]],axis=1)
dataset_test=pd.concat([dataset_test1,dataset_test["pickup_datetime"]],axis=1)
#determine corr
corr=dataset_int1.corr()
f,ax=plt.subplots(figsize=(7,5))             
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)                
      
#    """feature engineering"""
#1.we will derive new features from pickup_datetime variable
#new features will be year,month,day_of_week,hour
dataset_train1=pd.concat([dataset_int1,dataset_train["passenger_count"]],axis=1)
dataset_train2=pd.concat([dataset_train1,dataset_train["pickup_datetime"]],axis=1)          
                
#dataset_train2.isna().sum() 
data=[dataset_train2,dataset_test] 


for i in data:
    i["year"]=i["pickup_datetime"].apply(lambda row:row.year)
    i["month"]=i["pickup_datetime"].apply(lambda row:row.month)
    i["day_of_week"] = i["pickup_datetime"].apply(lambda row: row.dayofweek)
    i["hour"] = i["pickup_datetime"].apply(lambda row: row.hour)


# train2_nodummies=dataset_train2.copy()
# dataset_train2=train2_nodummies.copy()
    
plt.figure(figsize=(20,10))
sns.countplot(dataset_train2["year"])
# plt.savefig('year.png')

plt.figure(figsize=(20,10))
sns.countplot(dataset_train2['month'])
# plt.savefig('month.png')

plt.figure(figsize=(20,10))
sns.countplot(dataset_train2['day_of_week'])
# plt.savefig('day_of_week.png')

plt.figure(figsize=(20,10))
sns.countplot(dataset_train2['hour'])
# plt.savefig('hour.png')
plt.show

#Now we will use month,day_of_week,hour to derive new features like sessions in a day,seasons in a year,week:weekend/weekday

# for sessions in a day using hour columns

def f(x):
    if(x>=5) and (x<=11):
        return "morning"
    elif (x>=12) and (x<=16):
        return "afternoon"
    elif (x>=17) and (x<=20):
        return "evening"
    elif (x>=21) and (x<=23):
        return "night pm"
    elif (x>=0) and (x<=4):
        return "night am"

    
dataset_train2["sessions"]=dataset_train2["hour"].apply(f)    
dataset_test['session'] = dataset_test['hour'].apply(f)   

#for seasons in a year using month column   
        
def g(x):
    if (x>=3) and (x<=5):
        return "spring"
    elif (x>=6) and (x<=8):
        return "summer"
    elif (x>=9) and (x<=11):
        return "fall"
    else :
        return "winter"
        


dataset_train2['seasons'] = dataset_train2['month'].apply(g)
dataset_test['seasons'] = dataset_test['month'].apply(g)

#for week / weekend in a day of week columns
        
def h(x):
    if (x>=0) and (x<=4):
        return "weekday"
    elif (x>=5) and (x<=6):
        return "weekend"
       
    
dataset_train2['week'] = dataset_train2['day_of_week'].apply(h)
dataset_test['week'] = dataset_test['day_of_week'].apply(h)



dataset_train2['passenger_count'].describe()
dataset_train2.isnull().sum()
dataset_test.isna().sum()
#creating dummy varibale 

temp=pd.get_dummies(dataset_train2["passenger_count"],prefix="passenger_count")
dataset_train2=dataset_train2.join(temp)
temp = pd.get_dummies(dataset_test['passenger_count'], prefix = 'passenger_count')
dataset_test = dataset_test.join(temp)

temp = pd.get_dummies(dataset_test['seasons'], prefix = 'seasons')
dataset_test = dataset_test.join(temp)

temp=pd.get_dummies(dataset_train2["seasons"],prefix = "season" )
dataset_train2=pd.concat([dataset_train2,temp],axis=1)

temp = pd.get_dummies(dataset_train2['week'], prefix = 'week')
dataset_train2=pd.concat([dataset_train2,temp],axis=1)
temp = pd.get_dummies(dataset_test['week'], prefix = 'week')
dataset_test = dataset_test.join(temp)

temp = pd.get_dummies(dataset_train2['sessions'], prefix = 'sessions')
dataset_train2=pd.concat([dataset_train2,temp],axis=1)
temp = pd.get_dummies(dataset_test['session'], prefix = 'session')
dataset_test = dataset_test.join(temp)

temp = pd.get_dummies(dataset_train2['year'], prefix = 'year')
dataset_train2=pd.concat([dataset_train2,temp],axis=1)
temp = pd.get_dummies(dataset_test['year'], prefix = 'year')
dataset_test = dataset_test.join(temp)

#we will drop one column from each one-hot-encoded variables

dataset_train2.columns
dataset_test.columns
dataset_train2.info()

dataset_train2=dataset_train2.drop(['passenger_count_1.0','season_fall','week_weekday','sessions_afternoon','year_2009'],axis=1)
dataset_test=dataset_test.drop(['passenger_count_1','seasons_fall','week_weekday','session_afternoon','year_2009'],axis=1)

#3.Feature Engineering for latitude and longitude variable
#As we have latitude and longitude data for pickup and dropoff, we will find the distance the cab travelled from pickup and dropoff location.

#def haversine(coord1,coord2):
#    data=[dataset_train2,dataset_test]
#    for i in data:
#        lon1,lat1=coord1
#        lon2,lat2=coord2
#        r=6371000 #randius of earth in meters
#        phi_1=np.radians(i[lat1])
#        phi_2=np.radians(i[lat2])
#        delta_phi=np.radians(i[lat2]-i[lat1])
#        delta_lambda=np.radians(i[lon2]-i[lon1])
#        a=np.sin(delta_phi/2.0)**2+np.cos(phi_1)*np.cos(phi_2)*np.sin(delta_lambda/2.0)**2
#        c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
#        meters=c*r  #output distance in meter
#        km=meters/1000.0
#        miles=round(km,3)/1.609344
#        i["distance"]=miles
#    print(f"distance:{miles} miles")
#    return miles
#        
#haversine(['pickup_longitude','pickup_latitude'],['dropoff_longitude','dropoff_latitude'])        
        

#As Vincenty is more accurate than haversine. Also vincenty is prefered for short distances.
#Therefore we will drop great_circle. we will drop them together with other variables which were used to feature engineer.

from geopy.distance import geodesic
from geopy.distance import great_circle
#from sklearn.externals import joblib
data=[dataset_train2,dataset_test]
for i in data:
    i["great_circle"]=i.apply(lambda x : great_circle((x["pickup_latitude"],x["pickup_longitude"]),(x["dropoff_latitude"],x["dropoff_longitude"])).miles,axis=1)
    i["geodesic"]=i.apply(lambda x: geodesic((x["pickup_latitude"],x["pickup_longitude"]),(x["dropoff_latitude"],x["dropoff_longitude"])).miles,axis=1)
    

#We will remove the variables which were used to feature engineer new variable
    
dataset_train2=dataset_train2.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
       'month', 'day_of_week', 'hour', 'sessions', 'seasons', 'week','great_circle'],axis=1)    


dataset_test=dataset_test.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
       'month', 'day_of_week', 'hour', 'session', 'seasons', 'week','great_circle'],axis=1)

plt.figure(figsize=(20,5))
sns.boxplot(x=dataset_train2["geodesic"],data=dataset_train2,orient="h")

plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=dataset_train2['geodesic'],data=dataset_train2,orient='h')
plt.title('Boxplot of geodesic ')
# plt.savefig('bp geodesic.png')
plt.show()

dataset_train2.isnull().sum()

#outlier in geodesic

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

dataset_train11=pd.DataFrame(dataset_train2["geodesic"])

dataset_11=outlier_detect(dataset_train11)
dataset_train2=dataset_train2.drop(["geodesic"],axis=1)
dataset_train2=pd.concat([dataset_train2,dataset_11],axis=1)
dataset_train2.info()
#*****************************************************
#for test data
dataset_test1=pd.DataFrame(dataset_test["geodesic"])
dataset_test11=outlier_detect(dataset_test1)
dataset_test2=dataset_test.drop(["geodesic"],axis=1)
dataset_test=pd.concat([dataset_test2,dataset_test11],axis=1)
#**************************************************************
plt.boxplot(dataset_test["geodesic"])

dataset_train_num=["fare_amount","geodesic"]
dataset_train_obj=["passenger_count_2.0","passenger_count_3.0","passenger_count_4.0","passenger_count_5.0","passenger_count_6.0","season_spring","season_summer","season_winter","week_weekend","sessions_evening","sessions_morning","sessions_night am","sessions_night pm","year_2010","year_2011","year_2012","year_2013","year_2014","year_2015"]
len(dataset_train_obj)
dataset_train2[dataset_train_obj]=dataset_train2[dataset_train_obj].apply(lambda x: x.astype("category"))

dataset_test.info()

dataset_test_obj=["passenger_count_2","passenger_count_3","passenger_count_4","passenger_count_5","passenger_count_6","seasons_spring","seasons_summer","seasons_winter","week_weekend","session_evening","session_morning","session_night am","session_night pm","year_2010","year_2011","year_2012","year_2013","year_2014","year_2015"]
dataset_test[dataset_test_obj]=dataset_test[dataset_test_obj].apply(lambda x: x.astype("category"))
dataset_test.columns
#correlation
plt.figure(figsize=(15,15))
_=sns.heatmap(dataset_train2[dataset_train_num].corr(),square=True,cmap="RdYlGn",linewidth=0.5,linecolor="w",annot=True)
plt.savefig('correlation.png')# plt.savefig('correlation.png')

#As we can see from above correlation plot fare_amount and geodesic is correlated to each other.

#Jointplots for Bivariate Analysis.
#Here Scatter plot has regression line between 2 variables along with separate Bar plots of both variables.
#Also its annotated with pearson correlation coefficient and p value.
_=sns.jointplot(x="fare_amount",y="geodesic",data=dataset_train2,kind="reg")
_.annotate(stats.pearsonr)
plt.savefig('jointct.png')

#Chi-square test of Independence for Categorical Variables/Features
#Hypothesis testing :
#Null Hypothesis: 2 variables are independent.
#Alternate Hypothesis: 2 variables are not independent.
#If p-value is less than 0.05 then we reject the null hypothesis saying that 2 variables are dependent.
#And if p-value is greater than 0.05 then we accept the null hypothesis saying that 2 variables are independent.
#There should be no dependencies between Independent variables.
#So we will remove that variable whose p-value with other variable is low than 0.05.
#And we will keep that variable whose p-value with other variable is high than 0.05

#loop for chi2 test
for i in dataset_train_obj:
    for j in dataset_train_obj:
        if (i!=j):
            chi2,p,dof,ex=chi2_contingency(pd.crosstab(dataset_train2[i],dataset_train2[j]))
            if(p<0.05):
                print(i, "and ",j ,"are depandent to eath other",p,"---remove")
            else:
                print(i,"and",j,"are indepandent",p,"----keep")
                
#Analysis of Variance(Anova) Test
#it is carried out to comapre between each group in a categorical varibale
#ANOVA only lets us know the means for different groups are same or not. It doesn’t help us identify which mean is different.                
#Hypothesis testing :
#Null Hypothesis: mean of all categories in a variable are same.
#Alternate Hypothesis: mean of at least one category in a variable is different.
#If p-value is less than 0.05 then we reject the null hypothesis.
#And if p-value is greater than 0.05 then we accept the null hypothesis.
from statsmodels.formula.api import ols
dataset_train2.columns=["fare_amount","passenger_count_2","passenger_count_3","passenger_count_4","passenger_count_5","passenger_count_6","seasons_spring","seasons_summer","seasons_winter","week_weekend","session_evening","session_morning","session_night_am","session_night_pm","year_2010","year_2011","year_2012","year_2013","year_2014","year_2015","geodesic"]


import statsmodels.api as sm
model=ols("fare_amount ~ C(passenger_count_2)+C(passenger_count_3)+C(passenger_count_4)+C(passenger_count_5)+C(passenger_count_6)+C(seasons_spring)+C(seasons_summer)+C(seasons_winter)+C(week_weekend)+C(session_evening)+C(session_morning)+C(session_night_am)+C(session_night_pm)+C(year_2010)+C(year_2011)+C(year_2012)+C(year_2013)+C(year_2014)+C(year_2015)",data=dataset_train2).fit()
anova_table=sm.stats.anova_lm(model)

#Multicollinearity Test 

#VIF is always greater or equal to 1.
#if VIF is 1 --- Not correlated to any of the variables.
#if VIF is between 1-5 --- Moderately correlated.
#if VIF is above 5 --- Highly correlated.
#If there are multiple variables with VIF greater than 5, only remove the variable with the highest VIF.
from patsy import dmatrices 
from statsmodels.stats.outliers_influence import variance_inflation_factor
outcome,predictors=dmatrices("fare_amount ~ geodesic+passenger_count_2+passenger_count_3+passenger_count_4+passenger_count_5+passenger_count_6+seasons_spring+seasons_summer+seasons_winter+week_weekend+session_evening+session_morning+session_night_am+session_night_pm+year_2010+year_2011+year_2012+year_2013+year_2014+year_2015",dataset_train2,return_type="dataframe")
vif=pd.DataFrame()
vif["vif"]=[variance_inflation_factor(predictors.values,i) for i in range (predictors.shape[1])]
vif["features"]=predictors.columns
vif


#So we have no or very low multicollinearity

#feaure scalling with or with normlization
dataset_train2[dataset_train_num].var()
sns.distplot(dataset_train2["geodesic"],bins=50)

plt.figure()
stats.probplot(dataset_train2["geodesic"],dist="norm",fit=True,plot=plt)
#plt.savefig('qq prob plot.png')

#normlization

dataset_train2["geodesic"]=(dataset_train2["geodesic"]-min(dataset_train2["geodesic"]))/(max(dataset_train2["geodesic"])-min(dataset_train2["geodesic"]))

dataset_test['geodesic'] = (dataset_test['geodesic'] - min(dataset_test['geodesic']))/(max(dataset_test['geodesic']) - min(dataset_test['geodesic']))

dataset_train2["geodesic"].var()

sns.distplot(dataset_train2['geodesic'],bins=50)


plt.figure()
stats.probplot(dataset_train2["geodesic"],dist="norm",fit=True,plot=plt)
plt.savefig('qq prob plot1.png')


dataset_train2=dataset_train2.drop(["passenger_count_5","session_night_pm"],axis=1)
dataset_test=dataset_test.drop(["passenger_count_5","session_night pm"],axis=1)


#Splitting train into train and validation subsets¶
dataset_train2
x=dataset_train2.drop(["fare_amount"],axis=1)
y=dataset_train2["fare_amount"]

import sklearn
from sklearn.model_selection import train_test_split
#linesr regression

x_train, x_test,y_train, y_test =train_test_split(x,y,test_size=.2, random_state =100)
from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln=ln.fit(x_train,y_train)
ln.coef_
cofficent=pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(ln.coef_))],axis=1)



# Plot the coefficients
plt.figure(figsize=(15,5))
plt.plot(range(len(x_train.columns)), ln.coef_)
plt.xticks(range(len(x_train.columns)), x_train.columns, rotation=60)
plt.margins(0.02)
plt.savefig('linear coefficients')
plt.show()
ln.intercept_
y_pred=ln.predict(x_test)
y_error=y_test-y_pred

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
ss_residual=sum((y_test-y_pred)**2)
ss_total=sum((y_test-np.mean(y_test))**2)
r2_score=1-(ss_residual/ss_total)
r2_score

adjusted_r_squared= 1-(1-r2_score)*len((y_test)-1)/(len(y_test)-x.shape[1]-1)

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
rmse

#decision tree
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
model=dtr.fit(x_train,y_train)    
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

# random forest 
from sklearn.ensemble import RandomForestRegressor
rfc=RandomForestRegressor()
rfc_fit =rfc.fit(x_train,y_train)

y_pred=rfc_fit.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
rmse

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
random_grid={"n_estimators": range(100,500,100),"max_depth":range(5,20,1),"min_samples_leaf":range(2,5,1),"max_features":["auto","sqrt","log2"],"bootstrap":[True,False],"min_samples_leaf":range(2,5,1)}
forest_cv = RandomizedSearchCV(rfc, random_grid, cv=5)
forest=forest_cv.fit(x_train,y_train)
forest.best_estimator_


rfc_pred_train1 =forest.predict(x_train)
r2_score(y_train,rfc_pred_train1)

# prediction on test
rfc_pred1 = forest_cv.predict(x_test)
# test accuracy
r2_score(y_test,rfc_pred1)
# Random Search
rmse=np.sqrt(mean_squared_error(y_test,rfc_pred1))
rmse
scores=cross_val_score(forest,x_test,y_test,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def print_scores(scores):
    print("scores:", scores)
    print("mean:",scores.mean())
    print("std",scores.std())
    











































































   













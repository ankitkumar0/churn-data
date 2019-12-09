# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:43:09 2019

@author: Deepak Gupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("E:\\digit recognizer\\train.csv")
df.head(5)
y=df["label"]
x=df.drop("label",axis=1)
plt.plot(figsize=(7,7))
idx=100
grid_data=x.iloc[idx].as_matrix().reshape(28,28)  # 1d to 2d
plt.imshow(grid_data,interpolation="none",cmap="gray")
plt.show()

print(y[idx])
#PCA 
from sklearn.preprocessing import StandardScaler
standard_data=StandardScaler().fit_transform(x)
covar_matrix=np.matmul(standard_data.T,standard_data)
from scipy.linalg import eigh
values,vectors=eigh(covar_matrix,eigvals=(782,783))
vectors =vectors.T
new_cordinate=np.matmul(vectors,standard_data.T)
new_cordinates=np.vstack((new_cordinate,y)).T
dataframe=pd.DataFrame(data=new_cordinates,columns=("1st_principal","2nd_principal","labels"))
import seaborn as sns
sns.FacetGrid(dataframe,hue="labels",size=6).map(plt.scatter,"1st_principal","2nd_principal").add_legend()
# pca using scikit learn
from sklearn import decomposition
pca=decomposition.PCA()
pca.n_components=2
pca_data=pca.fit_transform(standard_data)
pca_data=np.vstack((pca_data.T,y)).T
pca_df=pd.DataFrame(data=pca_data,columns=("1st_principal","2nd_principal","labels"))
sns.FacetGrid(pca_df,hue="labels",size=6).map(plt.scatter,"1st_principal","2nd_principal").add_legend()

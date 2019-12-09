#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the data set from Desktop
dataset = pd.read_csv('ec2_cpu_utilization_53ea38.csv')

#changing the string to datetime
dataset['timestamp'] = pd.to_datetime(dataset.timestamp)

#changing the index of column
df = pd.DataFrame(dataset)
columnsTitles = ['timestamp', 'label', 'value']
dataset = df.reindex(columns=columnsTitles)


#splittiing the timestamp into year, month, day, hour and minute
dataset['year'] = dataset.timestamp.dt.year
dataset['month'] = dataset.timestamp.dt.month
dataset['day'] = dataset.timestamp.dt.day
dataset['hour'] = dataset.timestamp.dt.hour
dataset['minute'] = dataset.timestamp.dt.minute 
dataset.dtypes


# Changing type to categorical as they are int64
dataset['year'] = pd.Categorical(dataset['year'])
dataset['month'] = pd.Categorical(dataset['month'])
dataset['day'] = pd.Categorical(dataset['day'])
dataset['hour'] = pd.Categorical(dataset['hour'])
dataset['minute'] = pd.Categorical(dataset['minute'])
dataset.dtypes




# Dropping timestamp column as we don't need it now
dataset = dataset.drop('timestamp', axis = 1)

#X containing column [value, year, month, day, hour , minute]
x=dataset.drop('label', axis = 1)


#y containing column [label]
y=dataset.label



#import IsolationForest
from sklearn.ensemble import IsolationForest



# training the model
clf = IsolationForest(n_estimators=10, random_state=42)
clf.fit(x)

# prediction anomalies and normal values
y_pred = clf.predict(x)



# Array of predicted anomalies and normal 
y_pred




# Creating dataframe for anomalies and normal values for general analysis like value_counts
d = pd.DataFrame(y_pred, columns=['anomaly'])


#counting the values like number of 1 and number of -1
d.anomaly.value_counts()



# Percentage of anomalies
print("% of normal values", np.round(d.anomaly.value_counts()[1] * 100/ d.shape[0]), 3)
print("% of anomalies values", np.round(d.anomaly.value_counts()[-1] * 100/ d.shape[0]), 3)


#Note :- anomlies are outliers which is '-1' here

# Replacing 1 with 0 and -1 with 1
# 0 means normal and 1 means anomalies
d[d['anomaly'] == 1] = 0
d[d['anomaly'] == -1] = 1


#count the values , number of 0 and number of 1
d.anomaly.value_counts()




# Here we got anomalies (1) and normal (0) values
# We will not compare our predicted result with the provided label data
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

#F1 score
print("F1 score:", f1_score(d.anomaly, dataset.label))
#print("Accuracy score:", accuracy_score(d.anomaly, dataset.label))



#classification 
print("Classification report:")
print(classification_report(d.anomaly, dataset.label))



print("Confusion Matrix")
print(confusion_matrix(d.anomaly, dataset.label))




#####--------------Feature  Engineering------------------

#evaluating "value" variable and storing it in "val_3" column
x = dataset.drop('label', axis = 1)
x['val_3'] = x.value ** 3

#checking for the number of column x have
x.columns


# training the model
clf = IsolationForest(n_estimators = 10, random_state=42, behaviour='new')
clf.fit(x)

# prediction anomalies and normal values
y_pred = clf.predict(x)



# training the model
clf = IsolationForest(n_estimators = 10, random_state=42, behaviour='new')
clf.fit(x)

# prediction anomalies and normal values
y_pred = clf.predict(x)



# Creating dataframe for anomalies and normal values for general analysis like value_counts
d = pd.DataFrame(y_pred, columns=['anomaly'])

d.anomaly.value_counts()




# Percentage of anomalies
print("% of normal values", np.round(d.anomaly.value_counts()[1] * 100/ d.shape[0]), 3)
print("% of anomalies values", np.round(d.anomaly.value_counts()[-1] * 100/ d.shape[0]), 3)





# Replacing 1 with 0 and -1 with 1
# 0 means normal and 1 means anomalies

d[d['anomaly'] == 1] = 0
d[d['anomaly'] == -1] = 1




# Here we got anomalies (1) and normal (0) values
# We will not compare our predicted result with there provided label data

from sklearn.metrics import f1_score, confusion_matrix,  classification_report

print("F1 score for 0:", f1_score(d.anomaly, dataset.label, pos_label = 0))
print("F1 score for 1:", f1_score(d.anomaly, dataset.label, pos_label = 1))
#print("Accuracy score:", accuracy_score(d.anomaly, dataset.label))






print("Classification report:")
print(classification_report(d.anomaly, dataset.label))

print("Confusion Matrix")
print(confusion_matrix(d.anomaly, dataset.label))
















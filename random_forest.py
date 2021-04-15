# Importing necessary libraries

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

data = pd.read_csv("dataset/train.csv")
print(data.columns)
print(data["test preparation course"])

# Data Processing

X = data.drop(["Student_Id","race/ethnicity"],axis = 1)
print(X.head())

# Checking null Values

for i in list(X.columns):
    tab = "\t\t\t\t"
    if len(i)>8:
        tab = "\t\t\t"
    if len(i)>17:
        tab = "\t"
    print("{} {}: {}".format(i,tab,X[i].isnull().values.any()))
    
def process(data,columns = []):
    """
    @params : data = dataframe
    @params : 'columns' the columns to get dummies
    """
    
    for i in columns:
        category = pd.get_dummies(data[i])
        col = list(category.columns)[0]
        data = data.drop(i,axis = 1)
        data = pd.concat([data,category],axis = 1)
    return data
    
def dropColumns(data,columns = []):
    """
    @params : data = dataframe
    @params : 'columns' the columns to drop
    """
    return data.drop(columns,axis = 1)
    
Y = data["Pass/Fail"]
X = dropColumns(data,["Student_Id","race/ethnicity","Pass/Fail"])
print(data.shape)
print(X.shape)
print(Y.shape)

X = process(X,[["gender","parental level of education","test preparation course","lunch"]])
Y = pd.get_dummies(Y)
print(Y.sample(1))

# Training 

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state = 42,test_size = 0.1,stratify = Y)
print(x_train.shape)
print(y_train.shape)

# using RandomForestClassifier algorithm

forest = RandomForestClassifier(n_estimators = 1000)
forest.fit(x_train,y_train)

preds = forest.predict(x_test)

# Calculating the accuracy,f1score and recall

print(accuracy_score(y_test,preds))
print(classification_report(y_test,preds))

# Saving the model

filename = 'model.pkl'
pickle.dump(forest,open(filename,'wb'))
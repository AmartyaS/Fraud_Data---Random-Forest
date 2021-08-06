# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:08:20 2021

@author: ASUS
"""
#Importing all the Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Loading the Data File
file=pd.read_csv("D:\Data Science Assignments\Python-Assignment\Random Forest\Fraud_check.csv")
file.head()

#Data Manipulation
data=file
data.columns
data.describe()
data['Undergrad']=[1 if x=='YES' else 0 for x in file['Undergrad']]
data['Urban']=[1 if x=='YES' else 0 for x in file['Urban']]
data=pd.get_dummies(data,columns=["Marital.Status"],prefix=["Status"])
data['Taxable.Income']=["Risky" if x<=30000 else "Good" for x in file['Taxable.Income']]
data["Taxable.Income"].value_counts()
data.isna().sum()

#Initialising the target and the predictor variables
x=data[data.columns.difference(['Taxable.Income'])]
y=pd.DataFrame(data['Taxable.Income'])

#Splitting the dataframe into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#Making the model with 100 number of trees
model1=RandomForestClassifier(n_estimators=100)
model1.fit(x_train,y_train)
preds1=model1.predict(x_test)
#Checking the accuracy of the model
pd.Series(preds1).value_counts()
Accuracy1=metrics.accuracy_score(y_test,preds1)
Accuracy1


#Making the model with 10 number of trees
model2=RandomForestClassifier(n_estimators=10)
model2.fit(x_train,y_train)
preds2=model2.predict(x_test)
#Checking the accuracy of the model
pd.Series(preds2).value_counts()
Accuracy2=metrics.accuracy_score(y_test,preds2)
Accuracy2


#Making a model with collection of n_estimators to check the trend of its accuracy
a=[]
for i in range(1,100,4):
    model=RandomForestClassifier(n_estimators=i)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=metrics.accuracy_score(y_test,y_pred)
    a.append([i,acc])
accuracy=pd.DataFrame(a)
accuracy.columns=["N_Estimators","Accuracy Values"]
print(accuracy)

#Marking the important features of the dataframe
feature_imp=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_imp
#Visualizing the important features of the dataframe
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.legend()
plt.title("Visualizing Important Feature")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()


#Visualising the trend of accuracy of the model with increasing N_Estimator
plt.plot(accuracy.N_Estimators,accuracy['Accuracy Values'])
plt.title("Accuracy Vs No of Trees")
plt.xlabel("N_estimators")
plt.ylabel("Accuracy Scores")
plt.show()

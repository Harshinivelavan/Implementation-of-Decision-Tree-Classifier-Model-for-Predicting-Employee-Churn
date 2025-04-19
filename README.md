# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: V.HARSHINI
RegisterNumber:212224040109  
*/

import pandas as pd
 from sklearn.tree import DecisionTreeClassifier,plot_tree
 data=pd.read_csv("Employee (1).csv")
 data.head()
 data.info()
 data.isnull().sum()
 data["left"].value_counts()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data["salary"]=le.fit_transform(data["salary"])
 data.head()
 x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

 x.head()    
#no departments and no left
 y=data["left"]
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
 from sklearn.tree import DecisionTreeClassifier
 dt=DecisionTreeClassifier(criterion="entropy")
 dt.fit(x_train,y_train)
 y_pred=dt.predict(x_test)
 from sklearn import metrics
 accuracy=metrics.accuracy_score(y_test,y_pred)
 accuracy
 dt.predict([[0.5,0.8,9,260,6,0,1,2]])
 plt.figure(figsize=(8,6))
 plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True) 
 plt.show()
 
```

## Output:

![image](https://github.com/user-attachments/assets/bd50be10-9b94-480c-9ffb-04d50cf60581)



![Screenshot 2025-04-19 225857](https://github.com/user-attachments/assets/449aeef1-90f4-41ce-9bfc-853e64b78291)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

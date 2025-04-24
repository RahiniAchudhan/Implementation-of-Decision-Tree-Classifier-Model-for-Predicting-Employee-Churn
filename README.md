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

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAHINI A
RegisterNumber:  212223230165
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
# Data Head:
![image](https://github.com/user-attachments/assets/aef851ec-4435-4edf-9dad-2c0e78f4f8dc)

# Dataset Info:
![image](https://github.com/user-attachments/assets/30ce64b4-6da5-40c1-b3c6-59957be129a4)

# Null Dataset:
![image](https://github.com/user-attachments/assets/216604e0-e3af-4905-982d-090ce01480fa)

# Values Count in Left Column:
![image](https://github.com/user-attachments/assets/5e3ed06f-d425-4386-8eef-4b23a2e6ee91)

# Dataset transformed head:
![image](https://github.com/user-attachments/assets/cebdea2f-6339-4259-9dd7-a0956b49a257)

# X.head():
![image](https://github.com/user-attachments/assets/79cc32f8-1934-40ec-8b15-296eeecb33f6)

# Accuracy:
![image](https://github.com/user-attachments/assets/92cf7f3b-93bc-400e-b050-33c8f8ccdf29)

# Data Prediction:
![image](https://github.com/user-attachments/assets/e96519a5-c0de-4d91-8704-75091640ef8d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

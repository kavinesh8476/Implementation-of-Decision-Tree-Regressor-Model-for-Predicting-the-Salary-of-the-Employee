# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.

2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeRegressor , find MSE and R2 of the dataset.

5.Predict the values and end the program.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Kavinesh M
RegisterNumber:  212222230064
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### 1. data.head()
![head](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/5fdc6e33-91ec-44fe-9592-e221b2ece430)

### 2. data.info()
![info](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/c085ddd7-18dc-424a-962d-ba7537d06f96)

### 3. isnull() and sum()
![null](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/58717c61-714b-400b-beda-ce96ccd5977d)

### 4. data.head() for salary 
![data head](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/641c5a6b-34b8-4235-bb01-798a1410d9d6)

### 5. MSE value
![mse](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/d7a0b3df-bb2a-4d00-9c90-729ddcde8555)

### 6. r2 value
![r2](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/00ed3fac-2e33-4d45-a798-9067ccc63f99)

### 7. data prediction
![pred](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118466561/1732cbd7-76b6-4b4d-82d0-1511f7aab312)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

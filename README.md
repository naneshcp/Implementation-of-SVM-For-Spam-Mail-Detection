# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the spam email dataset (clean, tokenize, and vectorize the text data).
2. Split the dataset into training and testing sets.
3. Initialize the SVM model with appropriate kernel and parameters.
4. Train the SVM model using the training dataset.
5.Evaluate the model on the test dataset and classify emails as spam or not spam. 
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Naneshvaran
RegisterNumber: 24900972 
*/
import chardet
file=(r'C:\Users\admin\Downloads\spam.csv')
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r'C:\Users\admin\Downloads\spam.csv',encoding='Windows-1252')
print(data.head())
print(data.info())
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```

## Output:
![SVM For Spam Mail Detection](sam.png)

![output11(ml)](https://github.com/user-attachments/assets/0d8e737e-d9cf-4fb6-8616-c8d4d452b7b0)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 19:54:42 2018

@author: rohitraw
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy="mean", axis=0)
imp = imp.fit(X[:,0].reshape(-1, 1))
X[:, 0] = imp.transform(X[:,0].reshape(-1, 1))[:, 0]


# Encoding Categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])
OHE_X = OneHotEncoder(categorical_features=[0])
X = OHE_X.fit_transform(X).toarray()
y = LabelEncoder().fit_transform(y)
"""

# splitting data into train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

#feature scaling
# no need for feature scaling because sklearn will take care of it.
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# training the Linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

n = 0
errorPercent = 0

for inp,out in zip(X_test, y_test):
    res = regressor.predict([inp])
    #print(str(out) + " - " + str(res) + " = " + str(out - res[0]))
    errorPercent = errorPercent + (abs(out - res[0]) / out)
    n = n + 1
    
print("Accuracy = " + str(100 - (errorPercent/n)*100))



# Visualizing the training set
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Experience vs Salary - Training data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# Visualizing the testing set
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Experience vs Salary - Test data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Data Preprocessing Template

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
OHE_X = OneHotEncoder(categorical_features=[3])
X = OHE_X.fit_transform(X).toarray()

#Avoding Dummy Variable Trap (Although sklearn takes care of this by itself, the below line is just to be sure)
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Sklearn will do this too itself, atleast for regression)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

# Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


#Regression using column 0, 3 and 5 only
regressor_short = LinearRegression()
regressor_short.fit(X_train[:, [2,4]], y_train)
y_pred_short = regressor_short.predict(X_test[:, [2,4]])


from sklearn.metrics import mean_squared_error
print("Before -> ", mean_squared_error(y_test, y_pred))
print("After -> ", mean_squared_error(y_test, y_pred_short))

plt.scatter(range(len(y_test)), y_test, color="red")
plt.plot(range(len(y_pred_short)), y_pred_short, color="black")
plt.plot(range(len(y_pred)), y_pred, color="grey")
plt.show()

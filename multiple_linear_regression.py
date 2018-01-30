# Multiple Linear Regression

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding Categorical Data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Split data into Train and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Building an Optimal Model using Backwards Elimination
# Put 1's before X, axis for col=1, row(lines) = 0
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# X2 has highest P value, greater than SL(0.05) so remove it
X_opt = X[:, [0,1,3,4,5]]
# Fit model without X2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# X1 has highest P value, greater than SL(0.05) so remove it
X_opt = X[:, [0,3,4,5]]
# Fit model without X1
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# X2 has highest P value, greater than SL(0.05) so remove it
X_opt = X[:, [0,3,5]]
# Fit model without X2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# X2 has highest P value, greater than SL(0.05) so remove it
X_opt = X[:, [0,3]]
# Fit model without X2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
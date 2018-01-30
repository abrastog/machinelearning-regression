# Simple Linear Regression

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('NBA.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, :-1].values
print(X.shape)
print(y.shape)

# Splitting the dataset into Train and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# None

# Fit Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
X_test = X_test.reshape(-1, 1)
y_pred = regressor.predict(X_test)

# Visualizing the Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Height vs. Points Per Game')
plt.xlabel('Height')
plt.ylabel('Points Per Game')
plt.show()

# Visualizing the Test Set Result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Height vs. Points Per Game')
plt.xlabel('Height')
plt.ylabel('Points Per Game')
plt.show()
# Simple Linear Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splittig the dataset into Train and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
# None

# Fit Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Visualizing the Training Set Results
plt.scatter(X_train, y_train, color ='red') # Plotting the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # y-values are the prediction of the X_train years of experience
plt.title("Salary vs Experience (Training Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualizing the Test Set Results
plt.scatter(X_test, y_test, color ='red') # Plotting the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # y-values are the prediction of the X_train years of experience
plt.title("Salary vs Experience (Test Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

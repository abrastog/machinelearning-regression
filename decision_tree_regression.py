# Decision Tree Regression

# Importing dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Decision Tree Regressor to the dataset
# import a class
from sklearn.tree import DecisionTreeRegressor
# Create object
regressor = DecisionTreeRegressor(random_state = 0)
# Fit to regressor
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the Decision Tree Results (for higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

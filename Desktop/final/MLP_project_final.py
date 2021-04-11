import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv('housing.csv')

print(df)
print("\nBoston Housing dataset has {} data points with {} variables each.".format(*df.shape))

prices = df['MEDV']
# Minimum price of the data
minimum_price = np.amin(prices)

# Maximum price of the data
maximum_price = np.amax(prices)

# Mean price of the data
mean_price = np.mean(prices)

# Median price of the data
median_price = np.median(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("\nStatistics for Boston housing dataset:\n")
print("Minimum price: ${0:.2f}".format(minimum_price))
print("Maximum price: ${0:.2f}".format(maximum_price))
print("Mean price: ${0:.2f}".format(mean_price))
print("Median price ${0:.2f}".format(median_price))
print("Standard deviation of prices: ${0:.2f}".format(std_price))

# checking for linearity between dependent variable and independent variables
# MEDV Vs RM
plt.scatter(df['RM'], df['MEDV'], color='red')
plt.title('MEDV Vs RM', fontsize=14)
plt.xlabel('RM', fontsize=14)
plt.ylabel('MEDV', fontsize=14)
plt.grid(True)
plt.show()

# MEDV Vs PTRATIO
plt.scatter(df['PTRATIO'], df['MEDV'], color='green')
plt.title('MEDV Vs PTRATIO', fontsize=14)
plt.xlabel('PTRATIO', fontsize=14)
plt.ylabel('MEDV', fontsize=14)
plt.grid(True)
plt.show()

# MEDV Vs LSTAT
plt.scatter(df['LSTAT'], df['MEDV'], color='blue')
plt.title('MEDV Vs LSTAT', fontsize=14)
plt.xlabel('LSTAT', fontsize=14)
plt.ylabel('MEDV', fontsize=14)
plt.grid(True)
plt.show()

"""
# to find max and min values in each column
column = df["RM"]
max_value = column.max()
min_value = column.min()
print(min_value, ' - ', max_value)
"""
x = df[['RM', 'PTRATIO', 'LSTAT']]
y = df['MEDV']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

New_RM = int(input('enter the no.of rooms\n'))
New_PTRATIO = int(input('enter the no.of pupils\n'))
New_LSTAT = int(input('enter the percentage of lower status of population\n'))

temp = regr.predict([[New_RM, New_PTRATIO, New_LSTAT]])
print('Predicted House Price: $',int(temp))

# print(df)

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

df = pd.read_csv('housing.csv')
"""
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

x = sm.add_constant(x)  # adding a constant

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)
# print(df)

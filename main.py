# This is a Python script for Multiple (Multivariate) Linear Regression Machine Learning.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Convention alias for Seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

path_to_file = 'C:/Users/Cudlino/Documents/CUW/Spring 2024 Courses/Artificial Intelligence/Unit 5/auto_mpg_1983.csv'
df = pd.read_csv(path_to_file)
# print(df.head())
# print(df.shape)
# print(df.describe().round(2).T)

variables = ['cylinders', 'displacement', 'weight']

for var in variables:
    plt.figure()  # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression lin
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='mpg', data=df).set(title=f'Regression plot of {var} and MPG');
    # plt.show()

correlations = df[variables].corr()
# annot=True displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heat map of MPG Data - Pearson Correlations');
# plt.show()

y = df['mpg']
X = df[['cylinders', 'displacement', 'weight']]
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
# print(X.shape)
regressor = LinearRegression()
regressor.fit(X_train.values, y_train.values)
print('Intercept: ', regressor.intercept_)

feature_names = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients,
                              index = feature_names,
                              columns = ['Coefficient value'])
print('\n', coefficients_df)

y_pred = regressor.predict(X_test.values)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'\nMean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('\nR²:', r2)

print(regressor.score(X_train.values, y_train.values))

arr = [[8, 455, 3500]]
pred_mpg = regressor.predict(arr)
print('Predicted mpg:', pred_mpg)

y_hp = df['mpg']
X_hp = df[['cylinders', 'displacement', 'horsepower', 'weight']]
SEED = 42
X_train_hp, X_test_hp, y_train_hp, y_test_hp = train_test_split(X_hp, y_hp, test_size=0.2, random_state=SEED)
# print(X.shape)
regressor_hp = LinearRegression()
regressor_hp.fit(X_train_hp.values, y_train_hp.values)
print('Intercept: ', regressor_hp.intercept_)

feature_names_hp = X_hp.columns
model_coefficients_hp = regressor_hp.coef_

coefficients_df_hp = pd.DataFrame(data = model_coefficients_hp,
                              index = feature_names_hp,
                              columns = ['Coefficient value'])
print('\n', coefficients_df_hp)

y_pred_hp = regressor_hp.predict(X_test_hp.values)
results_hp = pd.DataFrame({'Actual': y_test_hp, 'Predicted': y_pred_hp})
print(results_hp)

mae = mean_absolute_error(y_test_hp, y_pred_hp)
mse = mean_squared_error(y_test_hp, y_pred_hp)
rmse = np.sqrt(mse)

print(f'\nMean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test_hp - y_pred_hp)**2)
actual_minus_actual_mean = sum((y_test_hp - y_test_hp.mean())**2)
r2_hp = 1 - actual_minus_predicted/actual_minus_actual_mean
print('\nR²:', r2_hp)

print(regressor_hp.score(X_train_hp.values, y_train_hp.values))
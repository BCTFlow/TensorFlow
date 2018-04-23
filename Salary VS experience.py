
import sys
sys._enablelegacywindowsfsencoding()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('‪C:\\Users\\hareg\\Desktop\\Spring 2018 Courses\\CSC 439 Special Topics in C++\\Salary VS Experience.py')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values 


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
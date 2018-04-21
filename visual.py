import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary VS Experience.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values # Dependent Variable/Target Values

#Will have to develop some algorithm/code right here.

# Visualizing the Training Set Results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set Results
# There is some code I have to push again put some comment here again.

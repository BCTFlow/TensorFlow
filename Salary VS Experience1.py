
import sys
sys._enablelegacywindowsfsencoding()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#with open('Salary VS Experience.csv', 'r') as csvFile:
    #reader = csv.reader(csvFile)
    #for row in reader:
        #print(row)
#csvFile.close()

dataset = pd.read_csv('Salary VS Experience.csv','r')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


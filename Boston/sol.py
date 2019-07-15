import pandas as pd
import numpy as np 
import matplotlib as mpl

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

import statsmodels.api as sm

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

X_train = boston[['LSTAT']]
Y = boston[['MEDV']]

# print(X_train)
# reg = LinearRegression().fit(X_train, Y)

# print(reg.intercept_)


X2 = sm.add_constant(X_train)
est = sm.OLS(Y, X2).fit()
print(est.summary())

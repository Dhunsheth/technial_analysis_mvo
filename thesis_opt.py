import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy
import cvxpy as cp

data = pd.read_csv('Data/data.csv', header=0, index_col=0)
data_val = data.values

factor_data = pd.read_csv('Data/factor_data.csv', header=0, index_col=0)
factor_data_val = factor_data.values

i = 2
y = 1
beta = ()
while i < 15:
    y_val = data_val[:,i]
    x_val = factor_data_val

    linreg = linear_model.LinearRegression()
    model = linreg.fit(x_val, y_val)
    beta = np.append(beta, np.ndarray.tolist(model.coef_[0:3]))
    beta = np.reshape(beta, (y,3))
    y = y + 1
    i = i + 3

factor_var = []
i = 0
while i <= 2:
    factor_var = np.append(factor_var, np.var(factor_data_val[:,i]))
    i = i+1

F = np.diag(factor_var)
Q = np.matmul(np.matmul(beta, F), np.transpose(beta))

i = 0
mu = ()
while i <= 4:
    mu_temp = beta[i,0]*np.average(factor_data_val[:,0]) + beta[i,1]*np.average(factor_data_val[:,1]) + beta[i,2]*np.average(factor_data_val[:,2]) + np.average(factor_data_val[:,3])
    mu = np.append(mu, mu_temp)
    i = i + 1

R = 0.2/252
# w_weights = []
# j = 0.01
# i = 0
# while i < 10:
w = cp.Variable(5)
ret = mu@w
risk = cp.quad_form(w, Q)
prob = cp.Problem(cp.Minimize(risk),
    [np.ones(5)@w == 1,
     w >= 0.05,
     w <= 0.45,
     ret >= R])
prob.solve()
w_weights = w.value
    # w_weights.append(w.value)
    # print(w.value)
    # j = j + 0.01
    # i = i + 1

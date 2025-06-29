# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder/'salary.csv', usecols=['Years of Experience', 'Salary'])
df.dropna(inplace=True)
X = df.drop('Salary', axis=1).to_numpy()
X_expanded = np.c_[np.ones((X.shape[0], 1)), X]
y = df.Salary.to_numpy()

# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# lr.fit(X, y)
# print(lr.coef_)
# print(lr.intercept_)
# [7046.76834403]
# 58283.27509417916

#%%
def calc_gradients(X, y, y_pred):
    e = y - y_pred
    n = X.shape[0]
    return -2 / n * X_expanded.T @ e


def predict(X, beta):
    return np.dot(X, beta)


# %%

beta = np.random.randn(X_expanded.shape[1])
lr = 0.001

for _ in range(1000):
    y_pred = predict(X_expanded, beta)
    gradients = calc_gradients(X_expanded, y, y_pred)
    beta = beta - lr * gradients

    plt.clf()
    plt.scatter(X[:,0], y, alpha=0.5)
    plt.plot(X[:,0], y_pred, color='blue')
    plt.title(f"$y = {beta[1]:.2f}\cdot x + {beta[0]:.2f}$")
    plt.pause(0.5)


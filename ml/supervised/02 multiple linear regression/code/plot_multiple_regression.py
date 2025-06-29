# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder/'salary.csv', usecols=['Age', 'Years of Experience', 'Salary'])
df.dropna(inplace=True)
df = df.head(500)
X = df.drop('Salary', axis=1).to_numpy()
y = df.Salary.to_numpy()
# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# coef: array([-1751.74391865,  9111.0842255 ])
# intercept: np.float64(100469.8008434252)
# %%
x0_range = np.linspace(X[:,0].min(), X[:,0].max(), 50)
x1_range = np.linspace(X[:,1].min(), X[:,1].max(), 50)
X0, X1 = np.meshgrid(x0_range, x1_range)
Z = lr.predict(np.c_[X0.ravel(), X1.ravel()]).reshape(X0.shape)

# %%
ax = plt.subplot(projection='3d', computed_zorder=False)

ax.scatter(X[:,0], X[:, 1], y, color='green', zorder=1, alpha=0.6)
ax.plot_surface(X0, X1, Z, zorder=0, alpha=0.4, color='blue')

ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Salary')
plt.show()
# %%

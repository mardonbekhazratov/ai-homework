import numpy as np
from typing import Literal

InverseFunc = Literal['pseudoinverse', 'inverse']

class LinearRegression:
    def __init__(self, inverse_func: InverseFunc='pseudoinverse'):
        self.coef_ = None
        self.intercept_ = None
        self.inverse_func = inverse_func

    def fit(self, X, y):
        X = self._add_dummy_variable(X)
        y = np.array(y)
        inv = self._get_inverse_function()
        self.weights_ = inv(X.T @ X) @ X.T @ y
        self.coef_ = self.weights_[1:]
        self.intercept_ = self.weights_[0]
        return self

    def predict(self, X):
        X = self._add_dummy_variable(X)
        return X @ self.weights_
    
    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def _get_inverse_function(self):
        match self.inverse_func:
            case 'pseudoinverse':
                return np.linalg.pinv
            case 'inverse':
                return np.linalg.inv
            case _:
                raise ValueError(self.inverse_func)
    
    @staticmethod
    def _add_dummy_variable(X):
        X = np.array(X)
        return np.c_[np.ones(X.shape[0]), X]
    
            
if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd

    folder = Path(__file__).resolve().parent
    df = pd.read_csv(folder/'salary.csv')
    df.dropna(inplace=True)
    X = df[['Age', 'Years of Experience']]
    y = df['Salary']

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    print(lin_reg.score(X, y))

    print('='*50)
    
    from sklearn.linear_model import LinearRegression as SKLinearRegression

    lin_reg = SKLinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    print(lin_reg.score(X, y))
    
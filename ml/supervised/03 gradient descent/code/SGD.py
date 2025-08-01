import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score

class StochasticGradientDescent:
    def __init__(self, lr, max_iter, tol):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.intercept_ = None

        self.lossi = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = X.shape[0]

        X = np.c_[np.ones(n), X]

        n_features = X.shape[1]

        coef = np.random.randn(n_features)

        converged = False

        for epoch in range(self.max_iter):
            for i in range(n):
                y_pred = X[i] @ coef
                e = y[i] - y_pred

                loss = root_mean_squared_error([y[i]], [y_pred])
                self.lossi.append(loss)

                coef_grad = (-2 / n) * X[i].T @ e

                if np.linalg.norm(self.lr * coef_grad) < self.tol:
                    converged = True
                    break

                coef = coef - self.lr * coef_grad

            
        if not converged:
            raise Exception(f"Did not converge after {self.max_iter} steps.")
        
        self.coef_ = coef[1:]
        self.intercept_ = coef[0]
        self.step = epoch

        return self
    
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ np.r_[self.intercept_, self.coef_]

    def score(self, X, y):
        return r2_score(y, self.predict(X))
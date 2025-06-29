import numpy as np


class Ridge:
    
    def __init__(self, alpha=1.0, inverse_func='pseudoinverse'):
        """
        Ridge Regression.

        Parameters:
        - alpha: Regularization strength (λ in Ridge formula). Must be non-negative.
        - inverse_func: Method for matrix inversion ('pseudoinverse' or 'inverse').
        """
        self.alpha = alpha
        self.inverse_func = inverse_func
        self.weights_ = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = self._add_dummy_variable(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # regularization matrix (λ * I, exclusing the intercept term)
        reg_matrix = self.alpha * np.eye(X.shape[1])
        reg_matrix[0, 0] = 0 # No regularization on the intercept

        # Compute weights using the Ridge formula
        inv_func = self._get_inverse_function()
        self.weights_ = inv_func(X.T @ X + reg_matrix) @ X.T @ y
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
        return self
    
    def predict(self, X):
        X = self._add_dummy_variable(X)
        return X @ self.weights_
    
    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        ss_total = np.sum((y - y.mean())**2)
        ss_residual = np.sum((y - y_pred)**2)
        return 1 - (ss_residual /  ss_total)
    
    def _get_inverse_function(self):
        match self.inverse_func:
            case 'pseudoinverse':
                return np.linalg.pinv
            case 'inverse':
                return np.linalg.inv
            case _:
                raise ValueError(f"Unknown inverse_func: {self.inverse_func}")

    @staticmethod
    def _add_dummy_variable(X):
        X = np.array(X)
        return np.c_[np.ones(X.shape[0]), X]
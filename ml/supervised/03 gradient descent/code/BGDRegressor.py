import numpy as np


class BGDRegressor:

    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights_ = None
        self.coef_ = None
        self.intercept_ = None
        self.loss_hisotry_ = []

    def fit(self, X, y):
        X = self._add_dummy_variable(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Initialize weights
        self.weights_ = np.random.randn(n_features)

        for epoch in range(self.max_iter):
            y_pred = X @ self.weights_
            error = y_pred - y

            # Compute gradient
            gradients = (1 / n_samples) * (X.T @ error)

            # Update weights
            self.weights_ -= self.learning_rate * gradients

            # Compute loss (Mean Squared Error)
            loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.loss_hisotry_.append(loss)

            # Check for convergence
            if epoch > 0 and abs(self.loss_hisotry_[-2] - loss) < self.tolerance:
                print(f"Converged after {epoch} epochs")
                break
        
        # Store coefficients and intercept separately
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
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
    
    @ staticmethod
    def _add_dummy_variable(X):
        X = np.array(X)
        return np.c_[np.ones(X.shape[0]), X]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = [[1], [2], [3], [4], [5]]
    y = [2.2, 2.8, 4.5, 3.7, 5.5]

    regressor = BGDRegressor(learning_rate=0.1, max_iter=1000)
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    # Model evaluation
    print(f"R² score: {regressor.score(X, y)}")
    print(f"Coefficients: {regressor.coef_}")
    print(f"Intercept: {regressor.intercept_}")

    plt.scatter([i[0] for i in X], y, alpha=0.6, s=10)
    plt.plot([i[0] for i in X], y_pred)
    plt.show()
import numpy as np

class SGDRegressor:
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights_ = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = self._add_dummy_variable(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Initialize weights
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights_ = np.random.randn(n_features)

        for epoch in range(self.max_iter):
            prev_weights = self.weights_.copy()
            for i in range(n_samples): # Process each sample one at a time
                x_i = X[i]
                y_i = y[i]
                gradients = 2 * x_i.T * (x_i @ self.weights_ - y_i)
                self.weights_ -= self.learning_rate * gradients

            # Check for convergence
            if np.linalg.norm(self.weights_ - prev_weights, ord=2) < self.tolerance:
                print(f'Converged in {epoch + 1} epochs.')
                break

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

    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])

    # Train SGD Regressor
    sgd = SGDRegressor(learning_rate=0.01, max_iter=1000, tolerance=1e-6, random_state=42)
    sgd.fit(X, y)

    # Results
    print("Intercept:", sgd.intercept_)
    print("Coefficients:", sgd.coef_)
    print("R² Score:", sgd.score(X, y))

    y_pred = sgd.predict(X)

    plt.scatter(X[:,0], y, alpha=0.6, s=10)
    plt.plot(X[:, 0], y_pred)
    plt.show()


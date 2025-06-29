import numpy as np

class MBGDRegressor:

    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=16, tolerance=1e-6, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights_ = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = self._add_dummy_variable(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights_ = np.random.rand(n_features)

        for epoch in range(self.max_iter):
            prev_weights = self.weights_.copy()
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # compute gradient for the mini batch
                errors = y_batch - X_batch @ self.weights_
                gradients = -2 * X_batch.T @ errors / len(y_batch)
                self.weights_ -= self.learning_rate * gradients

            if np.linalg.norm(self.weights_ -  prev_weights, ord=2) < self.tolerance:
                print(f'Converged in {epoch+1} epochs.')
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


    @staticmethod
    def _add_dummy_variable(X):
        X = np.array(X)
        return np.c_[np.ones(X.shape[0]), X]
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])

    # Train MBGD Regressor
    mbgd = MBGDRegressor(learning_rate=0.01, max_iter=1000, batch_size=2, tolerance=1e-6, random_state=42)
    mbgd.fit(X, y)

    # Results
    print("Intercept:", mbgd.intercept_)
    print("Coefficients:", mbgd.coef_)
    print("R² Score:", mbgd.score(X, y))

    y_pred = mbgd.predict(X)

    plt.scatter(X[:,0], y, alpha=0.6, s=10)
    plt.plot(X[:, 0], y_pred)
    plt.show()

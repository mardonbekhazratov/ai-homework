import numpy as np

class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, max_iter=100, tol=1e-4, solver='gradient_descent', learning_rate=0.01):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y, weights):
        m = len(y)
        logits = X @ weights
        h = self.sigmoid(logits)
        loss = (-1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        
        if self.penalty == 'l2':  # Ridge Regularization
            loss += (1 / (2 * self.C)) * np.sum(weights[1:] ** 2)  # Skip bias term

        return loss

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]  # Add bias term

        n_features = X.shape[1]
        weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            logits = X @ weights
            predictions = self.sigmoid(logits)
            gradient = (1 / len(y)) * (X.T @ (predictions - y))

            if self.penalty == 'l2':  # Apply L2 regularization
                gradient[1:] += (1 / self.C) * weights[1:]

            weights -= self.learning_rate * gradient

            if np.linalg.norm(gradient) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]  # Add bias term

        logits = X @ np.r_[self.intercept_, self.coef_] if self.fit_intercept else X @ self.coef_
        probs = self.sigmoid(logits)
        return np.c_[1 - probs, probs]  # Return class probabilities

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class SimpleClassifier:
    def __init__(self, lr=0.001, tol=1e-4, max_iter=1000):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.W = None
        self.b = None
    
    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    
    def cross_entropy(self, probs, y_true):
        n = probs.shape[0]
        return -(np.multiply(np.log(probs), y_true)).sum() / n

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n, n_features = X.shape
        y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        n_classes = y_one_hot.shape[1]

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        self.lossi = []

        for epoch in range(self.max_iter):
            logits = X @ self.W + self.b
            probs = self.softmax(logits)

            # Calculate loss
            loss = self.cross_entropy(probs, y_one_hot)
            self.lossi.append(loss)

            dW = X.T @ (probs - y_one_hot) / n
            db = np.sum(probs - y_one_hot, axis=0) / n

            self.W -= dW * self.lr
            self.b -= db * self.lr

            if np.linalg.norm(dW) < self.tol and abs(db) < self.tol:
                break

        return self
                           
    def predict(self, X):
        logits = X @ self.W + self.b
        probs = self.softmax(logits)

        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).sum() / len(y)
import numpy as np

class KMeans:

    cluster_centers_ = None
    inertia_ = None

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y = None):
        n_samples, n_features = X.shape
        self.cluster_centers_ = np.random.randn()
        for iter in range(self.max_iter):
            pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


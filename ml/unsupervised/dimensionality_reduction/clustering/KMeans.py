import numpy as np
import pandas as pd

class KMeans:

    cluster_centers_ = None
    inertia_ = None

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y = None):
        n_samples, n_features = X.shape
        self.cluster_centers_ = np.random.randn(self.n_clusters, n_features)
        for iter in range(self.max_iter):
            distances = np.stack([np.sqrt(np.sum((X - cluster_center)**2, axis=1)) \
                                  for cluster_center in self.cluster_centers_], axis=1)
            clusters = distances.argmin(axis=1)

            means = []

            for cluster in range(self.n_clusters):
                mean = X[clusters == cluster].mean(axis=0)
                means.append(mean)
            
            old_centroids = self.cluster_centers_
            self.cluster_centers_ = np.stack(means, axis=0)

            if np.sqrt(np.sum((self.cluster_centers_ - old_centroids)**2, axis=1)).max() < self.tol:
                self.n_iters_ = iter + 1
                break


    def predict(self, X):
        distances = np.stack([np.sqrt(np.sum((X - cluster_center)**2, axis=1)) \
                                  for cluster_center in self.cluster_centers_], axis=1)
        clusters = distances.argmin(axis=1)
        return clusters

    def score(self, X, y):
        return (self.predict(X) == y).sum() / X.shape[0]

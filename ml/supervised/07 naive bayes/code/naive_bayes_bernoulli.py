import numpy as np

class BernoulliNB:

    def __init__(self, alpha=1.):
        """
        Bernoulli Naive Bayes Classifier

        Parameters:
        alpha (float): Smoothing parameter to avoid zero probabilities (Laplace Smoothing)
        """
        self.alpha = alpha
        self.class_prior_ = None
        self.feature_probs_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the BernoulliNB model on training data.

        Parameters:
        X (numpy array): Binary feature matrix of shape (n_samples, n_features)
        y (numppy array): Target labels of shape (n_samples,)
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Compute prior probabilities P(C)
        self.class_prior_ = counts / n_samples

        # Compute likelihood P(X|C) using Laplace smoothing
        self.feature_probs_ = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.feature_probs_[idx] = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2*self.alpha)

    def predict_proba(self, X):
        """
        Compute the posterior probabilities for each class.

        Parameters:
        X (numpy array): Binary feature matrix of shape (n_samples, n_features)

        Returns:
        probs (numpy array): Probabilities for each class (n_samples, n_classes)
        """
        n_samples, n_features = X.shape
        log_probs = np.zeros((n_samples, len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            log_prior = np.log(self.class_prior_[idx])

            log_likelihood = (X * np.log(self.feature_probs_[idx])) + ((1-X)*np.log(1-self.feature_probs_[idx]))

            log_probs[:, idx] = log_prior + np.sum(log_likelihood, axis=1)

        probs = np.exp(log_probs)
        probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def predict(self, X):
        """
        Predict the class labels.

        Parameters:
        X (numpy array): Binary feature matrix of shape (n_samples, n_features)

        Returns:
        predictions (numpy array): Predicted class labels
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
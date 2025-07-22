import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        """
        Multinomial Naive Bayes Classifier.

        Parameters:
        alpha (float): Smoothing parameter (Laplace Smoothing).
        """
        self.alpha = alpha
        self.class_prior_ = None  # P(C)
        self.feature_probs_ = None  # P(X|C)
        self.classes_ = None  # Unique class labels
    
    def fit(self, X, y):
        """
        Train the model on training data.

        Parameters:
        X (numpy array): Feature matrix (word frequencies) of shape (n_samples, n_features)
        y (numpy array): Target labels of shape (n_samples,)
        """
        # Find unique classes
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Compute prior probabilities P(C)
        self.class_prior_ = counts / n_samples

        # Compute likelihood P(X|C) using Laplace smoothing
        self.feature_probs_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Select samples belonging to class c
            word_counts = np.sum(X_c, axis=0) + self.alpha  # Add alpha smoothing
            total_count = np.sum(word_counts)  # Sum of all words in class c
            self.feature_probs_[idx] = word_counts / total_count  # Normalize
        
    def predict_proba(self, X):
        """
        Compute the posterior probabilities for each class.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)

        Returns:
        probs (numpy array): Probabilities for each class (n_samples, n_classes)
        """
        n_samples, n_features = X.shape
        log_probs = np.zeros((n_samples, len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            # Log Prior
            log_prior = np.log(self.class_prior_[idx])

            # Log Likelihood: Multinomial distribution
            log_likelihood = np.sum(X * np.log(self.feature_probs_[idx]), axis=1)

            # Sum Prior and Likelihood
            log_probs[:, idx] = log_prior + log_likelihood
        
        # Convert log-probabilities to actual probabilities
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)  # Normalize
        return probs

    def predict(self, X):
        """
        Predict the class labels.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)

        Returns:
        predictions (numpy array): Predicted class labels
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

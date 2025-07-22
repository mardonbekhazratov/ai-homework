import numpy as np

class GaussianNB:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}
        self.classes = None

    def fit(self, X, y):
        """
        Fit the model by computing class priors, means, and variances.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)
        y (numpy array): Target labels of shape (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Compute class priors P(C_k)
        for c in self.classes:
            X_c = X[y == c] # Subset where class == c
            self.class_priors[c] = X_c.shape[0] / n_samples # P(C_k)
            self.class_means[c] = np.mean(X_c, axis=0) # (mean per feature)
            self.class_variances[c] = np.var(X_c, axis=0) # variance per feature

    def _gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian probability density function (PDF).

        Parameters:
        x (numpy array): The feature values
        mean (numpy array): Mean of the feature
        var (numpy array): Variance of the feature

        Returns:
        Probability density value
        """
        eps = 1e-9 # Small value to prevent division by zero
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean)**2) / (2 * (var + eps)))
        return coeff * exponent
    
    def _compute_likelihood(self, X, class_label):
        """
        Compute likelihood P(X | C_k) for a given class.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)
        class_label (int or str): Class label

        Returns:
        Likelihood values for each sample
        """

        mean = self.class_means[class_label]
        var = self.class_variances[class_label]
        likelihoods = self._gaussian_pdf(X, mean, var)
        return np.prod(likelihoods, axis=1) # Multiply across all features
    
    def predict_proba(self, X):
        """
        Compute the posterios probabilities P(C_k | X) for all classes.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)

        Returns:
        numpy array: Probabilities for each class, shape (n_samples, n_classes)
        """
        posteriors = []

        for c in self.classes:
            prior = self.class_priors[c] # P(C_k)
            likelihood = self._compute_likelihood(X, c) # P(X | C_k)
            posteriors.append(prior * likelihood) # P(C_k) * P(X | C_k)

        posteriors = np.array(posteriors).T # Shape (n_samples, n_classes)
        return posteriors / posteriors.sum(axis=1, keepdims=True) # Normalize
    

    def predict(self, X):
        """
        Predict the class labels for samples in X.

        Parameters:
        X (numpy array): Feature matrix of shape (n_samples, n_features)

        Returns:
        numpy array: Predicted class labels
        """
        posteriors = self.predict_proba(X)
        return self.classes[np.argmax(posteriors, axis=1)]
    

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    print(y_pred)
    print(y_test)
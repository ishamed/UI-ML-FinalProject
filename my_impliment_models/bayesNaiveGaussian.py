import numpy as np

class GaussianNaiveBayesFromScratch:
    def __init__(self , priors=None):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = priors

    def _log_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        if self.priors is None:
            self.priors = np.zeros(n_classes)
            calculate_priors = True
        else:
            calculate_priors = False

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            if calculate_priors:
                self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict_row(x) for x in X])

    def _predict_row(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(self._log_pdf(idx, x))
            posteriors.append(prior + posterior)

        return self.classes[np.argmax(posteriors)]


class GaussianNBWrapper:
    def __init__(self, priors=None):
        self.model = GaussianNaiveBayesFromScratch(priors=priors)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def get_model():
    return GaussianNBWrapper(priors=[0.5, 0.5])
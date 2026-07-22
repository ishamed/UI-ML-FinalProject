import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_signed = np.where(y <= 0, -1, 1)

        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        for _ in range(self.n_estimators):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1

                        error = sum(w[y_signed != predictions])

                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            EPS = 1e-10
            min_error = np.clip(min_error, EPS, 1 - EPS)

            clf.alpha = 0.5 * np.log((1.0 - min_error) / min_error)

            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y_signed * predictions)

            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)

        return np.where(np.sign(y_pred) <= 0, 0, 1)
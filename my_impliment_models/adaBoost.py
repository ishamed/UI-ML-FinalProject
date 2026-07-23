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
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_signed = np.where(y <= 0, -1, 1)

        n_pos = np.sum(y_signed == 1)
        n_neg = np.sum(y_signed == -1)
        w = np.where(y_signed == 1, 1.0 / (2 * n_pos), 1.0 / (2 * n_neg))

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

                        error = np.sum(w[y_signed != predictions])
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            if min_error >= 0.5:
                print("early stop")
                break

            EPS = 1e-10
            min_error = np.clip(min_error, EPS, 1 - EPS)

            clf.alpha = self.learning_rate * 0.5 * np.log((1.0 - min_error) / min_error)
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y_signed * predictions)

            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)

        return np.where(np.sign(y_pred) <= 0, 0, 1)

class AdaBoostCustomWrapper:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.model = AdaBoost(n_estimators=n_estimators , learning_rate=learning_rate)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def get_model():
    return AdaBoostCustomWrapper(n_estimators=50 , learning_rate=0.05)
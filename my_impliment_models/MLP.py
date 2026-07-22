import numpy as np

class MLP:
    def __init__(self, hidden_size=32, learning_rate=0.01, epochs=200):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs

        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None

    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        np.random.seed(42)
        self.W1 = np.random.randn(n_features, self.hidden_size) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, 1))

        for _ in range(self.epochs):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self._relu(Z1)

            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self._sigmoid(Z2)

            dZ2 = A2 - y

            dW2 = np.dot(A1.T, dZ2) / n_samples
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._relu_derivative(Z1)

            dW1 = np.dot(X.T, dZ1) / n_samples
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_proba(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return self._sigmoid(Z2).ravel()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
import numpy as np

class MLP:
    def __init__(self, hidden_size=32, learning_rate=0.01, epochs=200, batch_size=64):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

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

        rng = np.random.RandomState(42)

        self.W1 = rng.randn(n_features, self.hidden_size) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = rng.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, 1))

        self.loss_history = []

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            batches = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                current_batch_size = X_batch.shape[0]

                Z1 = np.dot(X_batch, self.W1) + self.b1
                A1 = self._relu(Z1)

                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self._sigmoid(Z2)

                batch_loss = -np.mean(y_batch * np.log(A2 + 1e-9) + (1 - y_batch) * np.log(1 - A2 + 1e-9))
                epoch_loss += batch_loss
                batches += 1

                dZ2 = A2 - y_batch

                dW2 = np.dot(A1.T, dZ2) / current_batch_size
                db2 = np.sum(dZ2, axis=0, keepdims=True) / current_batch_size

                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * self._relu_derivative(Z1)

                dW1 = np.dot(X_batch.T, dZ1) / current_batch_size
                db1 = np.sum(dZ1, axis=0, keepdims=True) / current_batch_size

                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            avg_epoch_loss = epoch_loss / batches
            self.loss_history.append(avg_epoch_loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_epoch_loss:.4f}")

    def predict_proba(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return self._sigmoid(Z2).ravel()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


class MLPCustomWrapper:
    def __init__(self, hidden_size=64, learning_rate=0.1, epochs=500, batch_size=64, threshold=0.35):
        self.model = MLP(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        self.threshold = threshold

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X, threshold=self.threshold)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def get_model():
    return MLPCustomWrapper(
        hidden_size=64,
        learning_rate=0.1,
        epochs=500,
        batch_size=64,
        threshold=0.45
    )
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
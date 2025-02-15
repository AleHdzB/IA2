import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, X):
        _, p = X.shape
        Y_est = np.zeros(p)
        for i in range(p):
            y_est = np.dot(self.w, X[:, i]) + self.b
            Y_est[i] = 1 if y_est >= 0 else 0
        return Y_est

    def fit(self, X, Y, epochs=100, lr=0.1):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += lr * (Y[i] - y_est) * X[:, i]
                self.b += lr * (Y[i] - y_est)

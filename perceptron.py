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

    def fit(self, X, Y, epochs=100, lr=0.1, solver = 'perceptron'):
        p = X.shape[1]

        if solver == 'SGD':
            for _ in range(epochs):
                for i in range(p):
                    y_est = np.dot(self.w, X[:, i]) + self.b
                    self.w += lr * (Y[:, i] - y_est) * X[:, i]
                    self.b += lr * (Y[:, i] - y_est)

        elif solver == 'BGD':
            for _ in range(epochs):
                Yest = self.predict(X)
                self.w += (lr / p) * ((Y - Yest) @ X.T).ravel()
                self.b += (lr / p) * np.sum(Y - Yest)

        elif solver == 'PInv':
            Xhat = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
            What = Y.reshape(1, -1) @ np.linalg.pinv(Xhat)
            self.b = What[0, 0]
            self.w = What[0, 1:]
        else:
            raise ValueError('Invalid solver')

import numpy as np
import matplotlib.pyplot as plt

class LinearNeuron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, X):
        Y_est = np.dot(self.w, X) + self.b
        return Y_est
    
    def fit(self, X, Y, epochs=50, lr=0.1, solver='PInv'):
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
            raise ValueError('Solver not found')

# Ejemplo

p = 200
x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 3 * np.random.randn(p)
plt.plot(x.ravel(), y.ravel(), 'b.')

neuron = LinearNeuron(1)
neuron.fit(x, y, solver='PInv')

# Plot the fitted line
x_line = np.linspace(-1.1, 1.1, 100).reshape(1, -1)
y_line = neuron.predict(x_line)
plt.plot(x_line.ravel(), y_line.ravel(), '--r')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Neuron Fit')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

class LinearNeuron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, X):
        Y_est = np.dot(self.w, X) + self.b
        return Y_est
    
    def batcher(self, X, Y, batch_size):
        p = X.shape[1]
        li, ui = 0, batch_size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + batch_size, ui + batch_size
            else:
                return None
            
    def MSE(self, X, Y):
        p = X.shape[1]
        Yest = self.predict(X)
        return (1/2*p) * np.sum((Y-Yest)**2)


    def fit (self, X, Y , epochs=50, lr=0.1, batch_size = 20):

        p = X.shape[1]
        error_history = []
        for _ in range (epochs):
            mini_batch = self.batcher(X,Y, batch_size)
            for mX, mY in mini_batch:
                Yest = self.predict(mX)
                self.w += (lr/p) * ((mY-Yest)@mX.T).ravel()
                self.b += (lr/p) * np.sum(mY-Yest)
            error_history.append(self.MSE(X,Y))
        return error_history

# Ejemplo----------------------------------------------------------------

p = 200
x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 3 * np.random.randn(p)
plt.plot(x.ravel(), y.ravel(), 'b.')

neuron = LinearNeuron(1)
history = neuron.fit(x, y, batch_size=1)


# Plot the fitted line
x_line = np.linspace(-1.1, 1.1, 100).reshape(1, -1)
y_line = neuron.predict(x_line)
plt.plot(x_line.ravel(), y_line.ravel(), '--r')
plt.figure()
plt.plot(history)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE History')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Neuron Fit')
plt.show()                

            
# def fibonacci():
#     a, b = 0, 1
#     while True:
#         yield a
#         a, b = b, a + b

# # Ejemplo de uso
# fib_gen = fibonacci()
# for _ in range(10):
#     print(next(fib_gen))
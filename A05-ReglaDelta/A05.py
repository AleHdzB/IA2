import numpy as np
import matplotlib.pyplot as plt

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def sigmoid(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

class Neuron:  # Cambiado el nombre de la clase a mayúscula para evitar conflicto
    def __init__(self, n_inputs, activation_function=linear, lr=0.01):
        self.w = -1 + 2 * np.random.rand(n_inputs)  # Usar rand en lugar de randn
        self.b = -1 + 2 * np.random.rand()
        self.lr = lr
        self.f = activation_function
    
    def predict(self, X):
        Z = np.dot(self.w, X) + self.b
        return self.f(Z)
    
    def train(self, X, Y, epochs=100):
        p = X.shape[1]
        for _ in range(epochs):
            Z = np.dot(self.w, X) + self.b
            Yest, dy = self.f(Z, derivative=True)
            lg = (Yest - Y) * dy  # Corrección en el cálculo del gradiente
            self.w -= (self.lr / p) * np.dot(lg, X.T).ravel()  # Usar -= para descenso de gradiente
            self.b -= (self.lr / p) * np.sum(lg)

# Ejemplo 1
example = 2


if example == 1:
    X = np.array([[0, 0, 1, 1],
                [0, 1, 0, 1]])
    Y = np.array([0, 0, 0, 1])

    neuron = Neuron(2, sigmoid, lr=0.1)  # Reducir learning rate a 0.1
    neuron.train(X, Y, epochs=10000)  # Aumentar épocas para asegurar convergencia

    # Verificar pesos finales
    print("Pesos finales:", neuron.w)
    print("Bias final:", neuron.b)

    # Graficar puntos
    plt.figure()
    for i in range(X.shape[1]):
        if Y[i] == 0:
            plt.plot(X[0, i], X[1, i], 'or')
        else:
            plt.plot(X[0, i], X[1, i], 'ob')

    # Graficar línea de decisión dentro de los límites del gráfico
    x_min, x_max = -0.5, 1.5
    w1, w2, b = neuron.w[0], neuron.w[1], neuron.b
    x_vals = np.array([x_min, x_max])
    y_vals = (-w1 * x_vals - b) / w2
    plt.plot(x_vals, y_vals, '--k')

    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Problema AND con Neurona')
    plt.grid(True)
    plt.show()

else:
    p = 100 
    x = - 1 + 2 * np.random.rand(p).reshape(1,-1) 
    y = - 18 * x + 6 + 2.5 * np.random.randn(p)
    plt.plot(x,y,'.b') 
     
    neuron = Neuron (1, linear, 0.1) 
    neuron.train(x, y, epochs=100) 
   
    xn = np.array([[-1, 1]]) 
    plt.plot(xn.ravel(), neuron.predict(xn), '--r')
    plt.show()
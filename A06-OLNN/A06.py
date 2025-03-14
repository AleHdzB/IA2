import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

def softmax(z, derivative=False):
    e = np.exp(z - np.max(z, axis=0))
    a = e / np.sum(e, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

class OLN:
    '''One Layer Neural Network'''
    def __init__(self, n_inputs, n_output, activation_function=logistic):
        self.W = np.random.randn(n_output, n_inputs) * 0.01
        self.b = np.zeros((n_output, 1))
        self.f = activation_function

    def predict(self, X):
        Z = np.dot(self.W, X) + self.b
        return self.f(Z)

    def fit(self, X, Y, lr=0.01, epochs=500):
        p = X.shape[1]
        for _ in range(epochs):
            # Propagate
            Z = np.dot(self.W, X) + self.b
            Yest, dY = self.f(Z, derivative=True)

            # Loss gradient
            lg = (Yest - Y) * dY  # Corregido el signo del gradiente

            # Update weights
            self.W -= (lr / p) * np.dot(lg, X.T)
            self.b -= (lr / p) * np.sum(lg, axis=1, keepdims=True)

def normalize_data(X):
    '''Normalize data to the range [0, 1]'''
    X_min = np.min(X, axis=1, keepdims=True)
    X_max = np.max(X, axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min)

def draw(X, Y, net):
    # Plot data points    
    colors = ['red', 'blue', 'green', 'purple']
    classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    plt.figure(figsize=(8, 6))
    
    # Graficar puntos
    for i in range(Y.shape[0]):
        plt.scatter(X[0, Y[i, :] == 1], X[1, Y[i, :] == 1], 
                    color=colors[i], label=classes[i], alpha=0.7)

    # Graficar líneas de decisión
    x_vals = np.linspace(0, 1, 100)
    for i in range(net.W.shape[0]):
        w1, w2 = net.W[i, 0], net.W[i, 1]
        b = net.b[i, 0]
        
        # Ecuación del límite de decisión: w1*x1 + w2*x2 + b = 0
        y_vals = (-w1 * x_vals - b) / w2
        
        # Filtrar valores dentro del rango visible
        valid = (y_vals >= -0.3) & (y_vals <= 1.3)
        plt.plot(x_vals[valid], y_vals[valid], 
                linestyle='--', color=colors[i], 
                label=f'Límite Clase {i+1}')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('x1 ')
    plt.ylabel('x2 ')
    plt.title('Clasificación Multiclase con Función Logística')
    # plt.legend()
    plt.grid(True)
    plt.show()


def MLP_multiclass_draw(X, Y, net):
    plt.figure(figsize=(8, 6))
    
    # Graficar puntos de datos
    colors = ['red', 'blue', 'green', 'purple']
    classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    for i in range(Y.shape[0]):
        plt.scatter(X[0, Y[i, :] == 1], X[1, Y[i, :] == 1], 
                    color=colors[i], label=classes[i], alpha=0.7)

    # Crear una malla para las regiones de decisión
    xmin, xmax = np.min(X[0, :]) - 0.5, np.max(X[0, :]) + 0.5
    ymin, ymax = np.min(X[1, :]) - 0.5, np.max(X[1, :]) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    data = np.vstack([xx.ravel(), yy.ravel()])

    # Predecir las probabilidades para cada punto en la malla
    zz = net.predict(data)
    zz = np.argmax(zz, axis=0)  # Clase con mayor probabilidad
    zz = zz.reshape(xx.shape)

    # Graficar las regiones de decisión
    plt.contourf(xx, yy, zz, alpha=0.3, levels=len(classes)-1, 
                 colors=colors, linestyles='dashed')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('One vs All con Softmax')
    # plt.legend()
    plt.grid(True)
    plt.show()


# Cargar dataset y entrenar
df = pd.read_csv('A06-OLNN/Dataset_A05.csv')
X = np.array(df[['x1', 'x2']]).T
Y = np.array(df[['y1', 'y2', 'y3', 'y4']]).T

# Normalizar datos primero para mejor convergencia
X_normalized = normalize_data(X)

#----------------------------------------------------PARTE 1----------------------------------------------------
# Entrenar la red con función logística (Multy Layer Classification) 
net = OLN(n_inputs=2, n_output=4, activation_function=logistic)
net.fit(X_normalized, Y, lr=0.1, epochs=5000)  # Parámetros ajustados

# Visualizar resultados
draw(X_normalized, Y, net)

#----------------------------------------------------PARTE 2----------------------------------------------------
# Entrenar la red con función softmax  (One vs All)

# Entrenar la red
net = OLN(n_inputs=2, n_output=4, activation_function=softmax)
net.fit(X, Y, lr=0.1, epochs=10000)

# Visualizar las regiones de decisión
MLP_multiclass_draw(X, Y, net)
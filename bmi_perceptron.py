import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def normalize_data(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

def train_and_plot_bmi():
    np.random.seed(42)
    n_samples = 100
    peso = np.random.uniform(40, 120, n_samples)
    altura = np.random.uniform(1.4, 2.0, n_samples)

    imc = peso / altura**2
    Y = (imc >= 25).astype(int)

    X = np.vstack((peso, altura))
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_norm = normalize_data(X, X_min, X_max)

    model = Perceptron(2)
    model.fit(X_norm, Y, epochs=100, lr=0.1)

    predictions = model.predict(X_norm)

    plt.scatter(X_norm[0, Y == 0], X_norm[1, Y == 0], c='blue', label='No sobrepeso')
    plt.scatter(X_norm[0, Y == 1], X_norm[1, Y == 1], c='red', label='Sobrepeso')

    w1, w2, b = model.w[0], model.w[1], model.b
    x_values = np.array([X_norm[0].min(), X_norm[0].max()])
    y_values = -(w1 * x_values + b) / w2
    plt.plot(x_values, y_values, '--k', label='Neurona')

    plt.xlabel('Peso normalizado')
    plt.ylabel('Altura normalizada')
    plt.title('Perceptrón - Sobrepeso')
    plt.legend()
    plt.grid(True)
    plt.show()

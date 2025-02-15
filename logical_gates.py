import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def draw_perceptron(model, ax):
    w1, w2, b = model.w[0], model.w[1], model.b
    ax.plot([-2, 2], [(1 / w2) * (-w1 * (-2) - b), (1 / w2) * (-w1 * (2) - b)], '--k')

def plot_results(X, Y, model, title, ax):
    _, p = X.shape
    for i in range(p):
        ax.plot(X[0, i], X[1, i], 'ob' if Y[i] == 1 else 'or')
    draw_perceptron(model, ax)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

def train_and_plot_gates():
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    gates = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, (gate, Y) in enumerate(gates.items()):
        model = Perceptron(2)
        model.fit(X, Y)
        plot_results(X, Y, model, f'Perceptrón - Compuerta {gate}', axs[i])

    plt.tight_layout()
    plt.show()

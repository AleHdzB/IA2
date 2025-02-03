import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, X):
        _, p = X.shape
        Y_est = np.zeros(p)
        for i in range(p):
            y_est = np.dot(self.w, X[:, i]) + self.b
            if y_est >= 0:
                Y_est[i] = 1
            else:
                Y_est[i] = 0
        return Y_est

    def fit(self, X, Y, epochs=100, lr=0.1):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += lr * (Y[i] - y_est) * X[:, i]
                self.b += lr * (Y[i] - y_est)

# Función para dibujar el perceptrón
def draw_perceptron(model, ax):
    w1, w2, b = model.w[0], model.w[1], model.b
    ax.plot([-2, 2], [(1/w2)*(-w1*(-2) - b), (1/w2)*(-w1*(2) - b)], '--k')

# Datos de entrada (las mismas para todas las compuertas)
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# Etiquetas para las compuertas AND, OR y XOR
Y_and = np.array([0, 0, 0, 1])  # Compuerta AND
Y_or = np.array([0, 1, 1, 1])   # Compuerta OR
Y_xor = np.array([0, 1, 1, 0])  # Compuerta XOR

# Crear y entrenar los perceptrones
model_and = Perceptron(2)
model_and.fit(X, Y_and)

model_or = Perceptron(2)
model_or.fit(X, Y_or)

model_xor = Perceptron(2)
model_xor.fit(X, Y_xor)

# Función para graficar los resultados
def plot_results(X, Y, model, title, ax):
    _, p = X.shape
    for i in range(p):
        if Y[i] == 0:
            ax.plot(X[0, i], X[1, i], 'or')
        else:
            ax.plot(X[0, i], X[1, i], 'ob')
    draw_perceptron(model, ax)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

# Crear una figura con 3 subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Graficar AND
plot_results(X, Y_and, model_and, 'Perceptrón - Compuerta AND', axs[0])

# Graficar OR
plot_results(X, Y_or, model_or, 'Perceptrón - Compuerta OR', axs[1])

# Graficar XOR
plot_results(X, Y_xor, model_xor, 'Perceptrón - Compuerta XOR', axs[2])

plt.tight_layout()
plt.show()


# Función para normalizar datos (min-max normalization)
def normalize_data(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

# 2.1 Crear datos aleatorios de personas (peso y altura)
np.random.seed(42)  # Para reproducibilidad
n_samples = 100
peso = np.random.uniform(40, 120, n_samples)  # Peso en kg (entre 40 y 120 kg)
altura = np.random.uniform(1.4, 2.0, n_samples)  # Altura en metros (entre 1.4 y 2.0 m)

# Calcular IMC y determinar sobrepeso (etiquetas)
imc = peso / altura**2
Y = (imc >= 25).astype(int)  # 1 si tiene sobrepeso, 0 si no

# 2.2 Normalizar los datos
X = np.vstack((peso, altura))  # Unir peso y altura en una matriz
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X_norm = normalize_data(X, X_min, X_max)

# 2.3 Entrenar el perceptrón
model = Perceptron(2)
model.fit(X_norm, Y, epochs=100, lr=0.1)


# Predecir con el perceptrón
predictions = model.predict(X_norm)

# Graficar los datos y la neurona
plt.scatter(X_norm[0, Y == 0], X_norm[1, Y == 0], c='blue', label='No sobrepeso')
plt.scatter(X_norm[0, Y == 1], X_norm[1, Y == 1], c='red', label='Sobrepeso')


# Dibujar la neurona
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
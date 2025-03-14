import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticNeuron:

    def __init__(self, n_inputs):
        self.w = -1 * 2*np.random.rand(n_inputs)
        self.b = -1 * 2*np.random.rand()

    def prediction_proba(self, x):
        Z = np.dot(self.w,X)+ self.b
        Yest = 1 / (1+np.exp(-Z)) # pylint: disable=runtime-warning
        return Yest
    
    def predict(self, X, umbral =0.5):
        Z = np.dot(self.w,X)+ self.b
        Yest = 1 / (1+np.exp(-Z))# pylint: disable=runtime-warning
        return 1 * (Yest >= umbral)
    def fit(self, X, Y, epochs=500, lr=0.01):
        p =  X.shape[1]
        for _ in range(epochs):
            Yest =  self.prediction_proba(X)
            self.w += (lr/p) * np.dot((Y-Yest), X.T).ravel()
            self.b += (lr/p) * np.sum(Y-Yest)


# Load dataset from CSV file
file_path = './diabetes.csv'
data = pd.read_csv(file_path)

X = np.array(data.drop(columns='Outcome')).T
Y = np.array(data['Outcome']).T


# Min-Max Normalization
X_min = X.min(axis=0)  # Mínimo de cada columna (característica)
X_max = X.max(axis=0)  # Máximo de cada columna (característica)

# Normalización con la fórmula (X - X_min) / (X_max - X_min)
X_normalized = (X - X_min) / (X_max - X_min)

# Crear una instancia de la neurona
neuron = LogisticNeuron(n_inputs=X_normalized.shape[0])

# Entrenar la neurona con los datos normalizados
neuron.fit(X_normalized, Y, epochs=500, lr=0.01)

# Hacer predicciones con los datos normalizados
predictions_proba = neuron.prediction_proba(X_normalized)
predictions = neuron.predict(X_normalized)


accuracy = np.mean(predictions == Y)*100
# Imprimir resultados
# print("Predicciones (probabilidades):", predictions_proba)
# print("Predicciones (clasificación):", predictions)
print("Precisión:", round(accuracy), "%")
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 *np.random.rand(n_inputs)
        self.b = -1 + 2 *np.random.rand()

    def predict(self , X):
        _, p = X.shape
        Y_est = np.zeros(p)
        for i in range(p):
            y_est = np.dot(self.w,X[:,i]) + self.b
            if y_est >= 0:
                Y_est[i] = 1
            else:
                Y_est[i] = 0
        return Y_est
        
    def fit(self, X, Y, epochs = 100, lr = 0.1):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1))
                self.w += lr * (Y[i] - y_est) * X[:,i]
                self.b += lr * (Y[i] - y_est)


# Ejemplo

def draw_perceptron(model):
    w1,w2,b  = model.w[0], model.w[1], model.b
    plt.plot([-2,2], [(1/w2)*(-w1*(-2) - b), (1/w2)*(-w1*(2) - b)], '--k')


model = Perceptron(2)

#Datos

X = np.array([[0,0,1,1],[0,1,0,1]])
# Y = np.array([0,1,1,1])  #OR gate
# Y = np.array([0,0,0,1]) #AND gate
Y = np.array([0,1,1,0]) #XOR gate  (no linealmente separable)

model.fit(X,Y)

#Dibujo

_, p = X.shape
for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0,i],X[1,i],'or')
    else:
        plt.plot(X[0,i],X[1,i],'ob')
        
plt.title('Perceptron')
plt.grid('on')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel('x1')
plt.ylabel('x2')   
draw_perceptron(model)
plt.show()
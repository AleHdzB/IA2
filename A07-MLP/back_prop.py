import numpy as np
from activations import *
from draw import *
import pandas as pd

class MLP ():
    def __init__(self, layer_dims,
                 hidden_activation=tanh,
                 output_activation=logistic):
        #Atributes
        self.n = layer_dims
        self.L = len(layer_dims) - 1
        self.W = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)

        #Initialize weights and biases
        for l in range(1, self.L + 1):
            self.W[l] = -1 + 2 * np.random.randn(self.n[l], self.n[l - 1])
            self.b[l] = -1 + 2 * np.random.randn(self.n[l], 1)
            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation
    def predict(self, X):
        A = X.copy()
        for l in range(1, self.L + 1):
            Z = self.W[l] @ A + self.b[l]
            A = self.f[l](Z)
        return A
    
    def fit(self, X, Y, lr = 0.01, epochs = 500):
        p = X.shape[1]
        for _ in range (epochs):
            #initialize containers
            A = [None] * (self.L + 1)
            dA = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)

            #Forward pas----------------------------------------------
            A[0] = X.copy()
            for l in range(1, self.L + 1):
                Z = self.W[l] @ A[l - 1] + self.b[l]
                A[l], dA[l] = self.f[l](Z, derivative = True)        
            
            #Backward pass---------------------------------------------
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] =  -( Y - A[l])*dA[l] 
                else:
                    lg[l] =  (self.W[l + 1].T @ lg[l + 1])*dA[l] 
            
            #Gradient Descent------------------------------------------
            for l in range(1, self.L + 1):
                self.W[l] -= (lr/p)*lg[l] @ A[l - 1].T
                self.b[l] -= (lr/p) * np.sum(lg[l])

#example
df = pd.read_csv('A07-MLP/XOR.csv')

X = np.array(df[['x1', 'x2']]).T
Y = np.array(df[["y"]]).T

net = MLP((2,20,1), output_activation=logistic)
net.fit(X,Y, lr=0.01, epochs=500)

MLP_binary_draw(X,Y,net)
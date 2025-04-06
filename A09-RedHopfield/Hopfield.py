import numpy as np

class Hopfield:
    def _hebbian(self, S):
        n,p = S.shape
        self.W += (1/self.n) * S @ S.T
        np.fill_diagonal(self.W, 0.0)
    
    def _pseudoinverse(self, S):
        n,p = S.shape
        self.W += S @ np.linalg.pinv(S)
        np.fill_diagonal(self.W, 0.0) 

    def _storkey(self, S):
        h = self.W @ S
        self.W += (1/self.n) *(S @ S.T - S @ h.T + h @ S.T)  
        np.fill_diagonal(self.W, 0.0)

    def __init__(self, neurons, mode = 'hebbian'):
        self.n = neurons
        self.W = np.zeros((self.n, self.n))
        self.modes = {'hebbian': self._hebbian,
                     'pinv': self._pseudoinverse,
                     'storkey': self._storkey}
        self.train = self.modes[mode]
        
    def predict(self, S, iterations = 1):
        for i in range (iterations):
            S = np.sign(self.W @ S)
        return S
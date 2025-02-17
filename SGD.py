import numpy as np
import matplotlib.pyplot as plt
from lineal_neuron import LinearNeuron


p = 200
x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 3 * np.random.randn(p)
plt.plot(x.ravel(), y.ravel(), 'b.')

neuron = LinearNeuron(1)
neuron.fit(x, y, solver='BGD')

# Plot the fitted line
x_line = np.linspace(-1.1, 1.1, 100).reshape(1, -1)
y_line = neuron.predict(x_line)
plt.plot(x_line.ravel(), y_line.ravel(), '--r')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Neuron Fit')
plt.show()
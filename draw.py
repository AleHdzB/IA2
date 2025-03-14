import numpy as np
import matplotlib.pyplot as plt


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
    plt.title('Regiones de Decisión con Softmax')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1, i], 'ro', markersize=9)
        else:
            plt.plot(X[0,i], X[1, i], 'bo', markersize=9)
    xmin, xmax = np.min(X[0,:])-0.5, np.max(X[0,:])+0.5
    ymin, ymax = np.min(X[1,:])-0.5, np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contour(xx,yy,zz, [0.5], colors='k',
                linestyles='--', linewidths=2)
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.show()    
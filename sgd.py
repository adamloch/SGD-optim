import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
import matplotlib.pyplot as pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from sklearn.utils import shuffle
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation
def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def activation(x):
    return np.max(x, 0)


def custommodel(inputs, weights):
    return activation(weights[0] * inputs[:, 0] - weights[1] * inputs[:, 0] + weights[1] * weights[0]* inputs[:, 1])


def MSE(y, predicted):
    return (y - predicted)**2


class Model:

    def __init__(self, weights=[0.1, 1.0], function=custommodel, loss=MSE):
        self.weights = np.array(weights)
        self.function = function
        self.loss = loss

    def calculate_loss(self, y, data, weights=None):
        try:
            predicted = self.predict(data, weights)
        except:
            predicted = self.predict(data, self.weights)

        return np.sum(self.loss(y, predicted))/len(y)

    def predict(self, inputs, weights=None):
        try:
            return self.function(inputs, weights)
        except:
            return self.function(inputs, self.weights)

    def grid_loss(self, y, data, weights=[]):
        result = np.empty(weights[0].shape)
        for i in range(weights[0].shape[0]):
            for j in range(weights[0].shape[1]):
                result[i, j] = self.calculate_loss(
                    y, data, [weights[0][i, j], weights[1][i, j]])/data.shape[0]
        return result

    def eval_numerical_gradient(self, y, data):
        """ 
        a naive implementation of numerical gradient of f at x
        - f should be a function that takes a single argument
        - x is the point (numpy array) to evaluate the gradient at
        """
        x = self.weights

        grad = np.zeros(x.shape)
        h = 0.00001

        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            old_value = x[ix]
            x[ix] = old_value + h  # increment by h
            fxh = self.calculate_loss(y, data, x)  # evalute f(x + h)
            x[ix] = old_value  # restore to previous value (very important!)
            x[ix] = old_value - h  # evalute f(x - h)
            fx_h = self.calculate_loss(y, data, x)
            x[ix] = old_value

            # compute the partial derivative
            grad[ix] = (fxh - fx_h) / (2*h)  # the slope
            it.iternext()  # step to next dimension

        return grad

    def init_random_weight(self):
        self.weights = np.random.rand(self.weights.shape[0])*10

    def update_weights(self, gradient, lr=0.001):
        self.weights = self.weights - lr * gradient


if __name__ == "__main__":
    # test purposes
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2,
                        cluster_std=2.5, random_state=21)

    model = Model(weights=[0, 0])
    W1 = np.linspace(0, 10, 50)
    W2 = np.linspace(0, 10, 50)
    W1, W2 = np.meshgrid(W1, W2)
    # z = model.grid_loss(y, X, [W1, W2])

    # fig = pylab.figure(figsize=(16, 7))
    # ax1 = fig.add_subplot(121, projection='3d')

    # ax1.plot_surface(W1, W2, z, cmap=cm.jet)
    # ax1.set_xlabel(r'$\theta^1$', fontsize=18)
    # ax1.set_ylabel(r'$\theta^2$', fontsize=18)

    # ax2 = fig.add_subplot(122)
    # ax2.contour(W1, W2, z, 128,  cmap=cm.jet)

    # pylab.suptitle('Contour Surface', fontsize=24)
    # pylab.show()

    model.init_random_weight()
    model.weights = np.array([9.89,9.99])
    def next_batch(X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])

    lossHistory = []

    allLoss = []
    allW0 = []
    allW1 = []
    nb_epochs = 100
    for epoch in range(nb_epochs):
        epochLoss = []
        X, y = shuffle(X, y, random_state=0)
        for (batchX, batchY) in next_batch(X, y, 6):
            preds = model.predict(batchX)
            loss = model.calculate_loss(batchY, batchX)
            allW0.append(model.weights[0])
            allW1.append(model.weights[1])
            epochLoss.append(loss/batchX.shape[0])
            allLoss.append(loss/batchX.shape[0])
            gradient = model.eval_numerical_gradient(batchY, batchX)
            model.update_weights(gradient, lr = 10e-6)
        lossHistory.append(np.average(epochLoss))

    z = model.grid_loss(y, X, [W1, W2])

    fig = pylab.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(221, projection='3d')

    ax1.plot_surface(W1, W2, z, cmap=cm.coolwarm)
    ax1.set_xlabel(r'$\theta^1$', fontsize=18)
    ax1.set_ylabel(r'$\theta^2$', fontsize=18)
    
    projected_loss = []
    print(X.shape, y.shape)
    for W0t, W1t in zip(allW0, allW1):
        projected_loss.append(model.calculate_loss(y,X,[W0t, W1t])/len(y)+1)
    #ax1.plot(allW0, allW1, projected_loss, color='b')
    
  
    ax2 = fig.add_subplot(222)
    ax2.contour(W1, W2, z, 128,  cmap=cm.jet)
    #ax2.plot(allW0, allW1, color='r')
    def animation_frame(nframe):
        ww = allW0[:nframe*16]
        www = allW1[:nframe*16]
        ax2.plot(ww, www, color='r')
        pl = projected_loss[:nframe*16]
        ax1.plot(ww, www, pl, color='b')
    anim = animation.FuncAnimation(fig, animation_frame, frames=int(len(allW0)/16))
    #sfreq = Slider(axfreq, 'Freq', 1, len(y), valinit=len(y), valstep=20)

    pylab.suptitle('Contour Surface', fontsize=24)

    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(0, nb_epochs), lossHistory)

    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(0, len(allLoss)), allLoss)

    

    pylab.show()

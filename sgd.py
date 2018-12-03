import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
import matplotlib.pyplot as pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from sklearn.utils import shuffle
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import matplotlib.animation as animation



glob_lr = '10e-6'
glob_batch = '16'
glob_init_w0 = '9.50'
glob_init_w1 = '9.50'
glob_nb_epoch = '100'

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
        #print(lr)
        self.weights = self.weights - lr * gradient

if __name__ == "__main__":
    # test purposes
    glob_lr = '10e-6'
    glob_batch = '16'
    glob_init_w0 = '9.50'
    glob_init_w1 = '9.50'
    glob_nb_epoch = '100'

    


    def calculate(lr, batch_size, init_w0, init_w1, nb_epochs):
        (X, y) = make_blobs(n_samples=400, n_features=2, centers=2,
                            cluster_std=2.5, random_state=21)

        model = Model(weights=[init_w0, init_w1])
        W1 = np.linspace(0, 10, 50)
        W2 = np.linspace(0, 10, 50)
        W1, W2 = np.meshgrid(W1, W2)
        batchSize = batch_size

        def next_batch(X, y, batchSize):
            for i in np.arange(0, X.shape[0], batchSize):
                yield (X[i:i + batchSize], y[i:i + batchSize])

        lossHistory = []

        allLoss = []
        allW0 = []
        allW1 = []
        #nb_epochs = 100
        for epoch in range(nb_epochs):
            epochLoss = []
            X, y = shuffle(X, y, random_state=0)
            for (batchX, batchY) in next_batch(X, y, batch_size):
                preds = model.predict(batchX)
                loss = model.calculate_loss(batchY, batchX)
                allW0.append(model.weights[0])
                allW1.append(model.weights[1])
                epochLoss.append(loss/batchX.shape[0])
                allLoss.append(loss/batchX.shape[0])
                gradient = model.eval_numerical_gradient(batchY, batchX)
                #print(lr)
                model.update_weights(gradient, lr = lr)
            lossHistory.append(np.average(epochLoss))

        z = model.grid_loss(y, X, [W1, W2])
        print('policzono')
        return z, allLoss, lossHistory, allW0, allW1, W1, W2, X, y, model


    lr_val = float(glob_lr)
    batch_val = int(glob_batch)
    init_w0_val = float(glob_init_w0)
    init_w1_val = float(glob_init_w1)
    nb_epoch_val = int(glob_nb_epoch)
    z, allLoss, lossHistory, allW0, allW1, W1, W2, X, y, model = calculate(lr = lr_val, batch_size = batch_val, init_w0 = init_w0_val, init_w1 = init_w1_val, nb_epochs = nb_epoch_val)


    fig = pylab.figure(figsize=(16, 7))

    def actsa(event):
        
        print(glob_lr)
        lr_val = float(glob_lr)
        batch_val = int(glob_batch)
        init_w0_val = float(glob_init_w0)
        init_w1_val = float(glob_init_w1)
        nb_epoch_val = int(glob_nb_epoch)
        print(lr_val)
        print(batch_val)
        z, allLoss, lossHistory, allW0, allW1, W1, W2, X, y, model = calculate(lr = lr_val, batch_size = batch_val, init_w0 = init_w0_val, init_w1 = init_w1_val, nb_epochs = nb_epoch_val)
        
        pylab.clf()
        axbox = plt.axes([0.91, 0.35, 0.05, 0.045])
        text_box = TextBox(axbox, 'Learning Rate', initial=glob_lr)
        def submit(text):
            global glob_lr
            glob_lr = text
        text_box.on_submit(submit)

        bbox = plt.axes([0.77, 0.35, 0.05, 0.045])
        btext_box = TextBox(bbox, 'Batch Size', initial=glob_batch)
        def submit(text):
            global glob_batch
            glob_batch = text
        btext_box.on_submit(submit)

        w0box = plt.axes([0.77, 0.25, 0.05, 0.045])
        w0text_box = TextBox(w0box, 'Initial W0', initial=glob_init_w0)
        def submit(text):
            global glob_init_w0
            glob_init_w0 = text
        w0text_box.on_submit(submit)

        w1box = plt.axes([0.77, 0.155, 0.05, 0.045])
        w1text_box = TextBox(w1box, 'Initial W1', initial=glob_init_w1)
        def submit(text):
            global glob_init_w1
            glob_init_w1 = text
        w1text_box.on_submit(submit)

        nbbox = plt.axes([0.91, 0.25, 0.05, 0.045])
        nbtext_box = TextBox(nbbox, 'Number of epochs', initial=glob_nb_epoch)
        def submit(text):
            global glob_nb_epoch
            glob_nb_epoch = text
        nbtext_box.on_submit(submit)

        action = plt.axes([0.91, 0.05, 0.05, 0.045])
        bnext = Button(action, 'Next')
        bnext.on_clicked(actsa)
        ax1 = fig.add_subplot(231, projection='3d')
        
        ax1.plot_surface(W1, W2, z, cmap=cm.coolwarm)
        ax1.set_xlabel(r'$\theta^1$', fontsize=16)
        ax1.set_ylabel(r'$\theta^2$', fontsize=16)
        
        projected_loss = []
        print(X.shape, y.shape)
        for W0t, W1t in zip(allW0, allW1):
            projected_loss.append(model.calculate_loss(y,X,[W0t, W1t])/len(y)+1)
        
    
        ax2 = fig.add_subplot(232)
        c = ax2.contour(W1, W2, z, 128,  cmap=cm.coolwarm)
        def animation_frame(nframe):
            ww = allW0[:nframe*16]
            www = allW1[:nframe*16]
            ax2.plot(ww, www, color='r')
            pl = projected_loss[:nframe*16]
            ax1.plot(ww, www, pl, color='r')
        anim = animation.FuncAnimation(fig, animation_frame, frames=int(len(allW0)/16))
        pylab.colorbar(c)

        pylab.suptitle('Contour Surface', fontsize=24)

        ax3 = fig.add_subplot(235)
        ax3.plot(np.arange(0, len(lossHistory)), lossHistory)
        ax3.set_xlabel("Number of epoches")
        ax3.set_ylabel("Loss value")

        ax4 = fig.add_subplot(234)
        ax4.plot(np.arange(0, len(allLoss)), allLoss)
        pylab.show()

   
    #def s
    lrbox = plt.axes([0.91, 0.35, 0.05, 0.045])
    text_box = TextBox(lrbox, 'Learning Rate', initial=glob_lr)
    def submit(text):
        global glob_lr
        glob_lr = text
    text_box.on_submit(submit)

    bbox = plt.axes([0.77, 0.35, 0.05, 0.045])
    btext_box = TextBox(bbox, 'Batch Size', initial=glob_batch)
    def submit(text):
        global glob_batch
        glob_batch = text
    btext_box.on_submit(submit)

    w0box = plt.axes([0.77, 0.25, 0.05, 0.045])
    w0text_box = TextBox(w0box, 'Initial W0', initial=glob_init_w0)
    def submit(text):
        global glob_init_w0
        glob_init_w0 = text
    w0text_box.on_submit(submit)

    w1box = plt.axes([0.77, 0.155, 0.05, 0.045])
    w1text_box = TextBox(w1box, 'Initial W1', initial=glob_init_w1)
    def submit(text):
        global glob_init_w1
        glob_init_w1 = text
    w1text_box.on_submit(submit)

    nbbox = plt.axes([0.91, 0.25, 0.05, 0.045])
    nbtext_box = TextBox(nbbox, 'Number of epochs', initial=glob_nb_epoch)
    def submit(text):
        global glob_nb_epoch
        glob_nb_epoch = text
    nbtext_box.on_submit(submit)

    #text_box.on_submit(submit)
    action = plt.axes([0.91, 0.05, 0.05, 0.045])
    bnext = Button(action, 'Next')
    bnext.on_clicked(actsa)
    pylab.show()

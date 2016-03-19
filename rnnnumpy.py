import numpy as np
from datetime import datetime
import sys

class RNNNumpy:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        # Assign instance variables
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./wordDim), np.sqrt(1./wordDim), (hiddenDim, wordDim))
        self.V = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (wordDim, hiddenDim))
        self.W = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim, hiddenDim))
        
    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)
        
    def forwardPropagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because we need them later
        # We add one additional element for the initial hidden state, which we set to 0
        s = np.zeros((T + 1, self.hiddenDim))
        s[-1] = np.zeros(self.hiddenDim)
        # The outputs at each time step, save them for later
        o = np.zeros((T, self.wordDim))
        # For each time step
        for t in np.arange(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with the one-hot vector
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
        
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forwardPropagation(x)
        return np.argmax(o, axis=1)
    
    def calculateTotalLoss(self, x, y):
        L = 0
        # For each sentence
        for i in np.arange(len(y)):
            o, s = self.forwardPropagation(x[i])
            # We only care about our prediction of the "correct" words
            correctWordPredictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correctWordPredictions))
        return L
    
    def calculateLoss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculateTotalLoss(x, y)/N
        
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forwardPropagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        deltaO = o
        deltaO[np.arange(len(y)), y] -= 1
        # For each output backwards
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(deltaO[t], s[t].T)
            # Initial delta calculation
            deltaT = self.V.T.dot(deltaO[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bpttTruncate steps)
            for bpttStep in np.arange(max(0, t-self.bpttTruncate), t+1)[::-1]:
                dLdW += np.outer(deltaT, s[bpttStep-1])
                dLdU[:, x[bpttStep]] += deltaT
                # Update delta for the next step
                deltaT = self.W.T.dot(deltaT) * (1 - s[bpttStep-1] ** 2)
        return [dLdU, dLdV, dLdW]
        
    def gradient_check(self, x, y, h=0.001, errorThreshold=0.01):
        # Calculate the gradients using backpropagation. We want to check if these are correct
        bpttGradients = self.bptt(x, y)
        # List of all parameters we want to check
        modelParameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(modelParameters):
            # Get the actual parameter value from the model
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can restore it later
                originalValue = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = originalValue + h
                gradplus = self.calculateTotalLoss([x], [y])
                parameter[ix] = originalValue - h
                gradminus = self.calculateTotalLoss([x], [y])
                estimatedGradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to the original value
                parameter[ix] = originalValue
                # The gradient for this parameter calculated using backpropagation
                backpropGradient = bpttGradients[pidx][ix]
                # Calculate the relative error: (|x-y|/(|x|+|y|))
                relativeError = np.abs(backpropGradient - estimatedGradient)/(np.abs(backpropGradient) + np.abs(estimatedGradient))
                # If the error is too large, fail the gradient check
                if relativeError > errorThreshold:
                    print "Gradient check error: parameter=%s is=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimatedGradient
                    print "Backpropagation gradient: %f" % backpropGradient
                    print "Relative Error: %f" % relativeError
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)
            
    def sgdStep(self, x, y, learningRate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change the parameters according to gradients and learning rate
        self.U -= learningRate * dLdU
        self.V -= learningRate * dLdV
        self.W -= learningRate * dLdW
        
    def trainWithSgd(self, X_train, Y_train, learningRate=0.005, nepoch=100, evaluateLossAfter=5):
        # We keep track of the losses so we can plot them later
        losses = []
        numExamplesSeen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluateLossAfter == 0):
                loss = self.calculateLoss(X_train, Y_train)
                losses.append((numExamplesSeen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, numExamplesSeen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5 
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            # For each training examples
            for i in range(len(Y_train)):
                # One SGD step
                self.sgdStep(X_train[i], Y_train[i], learningRate)
                numExamplesSeen += 1
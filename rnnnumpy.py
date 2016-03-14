import numpy as np

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
        dldV = np.zeros(self.V.shape)
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
                print "Backpropagation step t=%d bptt step=%d "% (t, bpttStep)
                dLdW += np.outer(deltaT, s[bpttStep-1])
                dLdU[:, x[bpttStep]] += deltaT
                # Update delta for the next step
                deltaT = self.W.T.dot(deltaT) * (1 - s[bpttStep-1] ** 2)
        return [dLdU, dLdV, dLdW]
# Re-creation of author's code
import numpy as np
import random

# He keeps biases separate. We could wrap them up if we appended
# a constant 1 to each layer.
class Network(object):
    
    def __init__(self, sizes):
        # Saves off the sizes and the number of layers
        # Initializes all weights and biases
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size, 1) for size in self.sizes[1:]]
        self.weights = [np.random.randn(curr,prev) for prev, curr in zip(sizes[:-1], sizes[1:])]
        print(self.weights)

    def feedforward(self, a):
        ''' given self.sizes[0] input activations, computes and returns the output.  
        It throws away partial results.
        Needs to do sigmoid(wx) for each layer. 
        '''
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b) # Why dot and not matmult?
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data=None):
        ''' Stochastic gradient descent. We randomize and break the training set down into minibatches.
        We do num_epochs passes through the set. Each pass does a feedforward pass and a backprop
        pass to compute deltas. Eta is the learning rate. Training data is a list of tuples (x,y).
        If validation_data (same format) is provided, it will eval it and print out after each epoch.  
        '''
        if validation_data:
            n_val = len(list(validation_data))
        n = len(list(training_data))

        for epoch in range(epochs):
            # Randomize training data, break into mini-batches (could do outside loop instead)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
               self.update_mini_batch(mini_batch, eta)
            if validation_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(validation_data), n_val)) 
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, eta):
        ''' Computes the diffs for each weight and bias using the average error on a single pass. 
        
        '''
        dWeights = [np.zeros(weight.shape) for weight in self.weights]
        dBiases = [np.zeros(bias.shape) for bias in self.biases]
        for x, y in mini_batch:
            # do backprop on that and modify every dBias and dWeight 
            dWeightsOn1, dBiasesOn1 = self.backprop(x,y)
            dWeights = [dw+dw1 for dw, dw1 in zip(dWeights, dWeightsOn1)]
            dBiases = [db+db1 for db, db1 in zip(dBiases, dBiasesOn1)]

        # Turn into averages and add to weights and biases to take an SGD step.
        self.weights = [w-eta/len(mini_batch)*dw for w, dw in zip(self.weights, dWeights)] 
        self.biases = [b-eta/len(mini_batch)*db for b, db in zip(self.biases, dBiases)]

    def backprop(self, x, y):
        ''' Calculates and returns the delta for each weight and bias per error on this 
        single sample (x,y). The deltas returned have the same shape as self.weights and self.biases.
        This is the heavy lifting, and heart of the algorithm.   
        '''
        dWeights = [np.zeros(weight.shape) for weight in self.weights]
        dBiases = [np.zeros(bias.shape) for bias in self.biases]
        # forward
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward
        delta = self.cost_derivative(activations[-1], y) * dSigmoid(zs[-1])
        dWeights[-1] = np.dot(delta, activations[-2].T)
        dBiases[-1] = delta
        for L in range(2, self.num_layers):
            weight = self.weights[-L+1]
            delta = np.dot(weight.T, delta) * dSigmoid(zs[-L])
            dWeights[-L] = np.dot(delta, activations[-L-1].T) 
            dBiases[-L] = delta
        return dWeights, dBiases

    def cost_derivative(self, output_activations, y):
        return (output_activations - y) # why as a tuple?
        
    def evaluate(self, data):
        ''' Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.'''
        # test_results = sum(np.argmax(self.feedforward(x)) == y for x,y in data)
        test_results = sum(np.round(self.feedforward(x)) == y for x,y in data)
        return test_results

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def make_xor_training_data():
    a1 = (np.array([[0],[0]]), np.array([[0]]))
    a2 = (np.array([[0],[1]]), np.array([[1]]))
    a3 = (np.array([[1],[0]]), np.array([[1]]))
    a4 = (np.array([[1],[1]]), np.array([[0]]))
    return [a1, a2, a3, a4]

def random_xor_sample():
    x1 = random.randint(0,1)
    x2 = random.randint(0,1)
    y = x1 ^ x2
    return ( np.array([[x1], [x2]]), np.array([[y]]) )     

def make_xor_data(num):
    training_data = make_xor_training_data()
    validation_set = [random_xor_sample() for k in range(num)]
    test_set = [random_xor_sample() for k in range(num)]
    return training_data, validation_set, test_set

def main():
    training_data, validation_data, test_data = make_xor_data(10)
    # print(validation_data)

    net = Network([2, 2, 1])
    net.SGD(training_data, epochs=500, mini_batch_size=10, eta=3.0, validation_data=validation_data)
    print(net.weights)

main()

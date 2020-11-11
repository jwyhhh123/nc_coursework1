
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)
def sigmoid_d(x):
    return np.exp(x) / ((np.exp(x) + 1)**2)
def relu(x):
    return np.maximum(0,x)
def relu_d(x):
    r = x
    r[r<=0]=0
    r[r>0]=1
    return r
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = x - 0.5      # Center the input values between [-0.5,0.5]

        i = 0
        layer = 1
        while layer < self.L:
            self.z[layer] = np.dot(self.w[layer], self.a[i]) + self.b[layer]
            #self.z[layer] = self.z[layer]/np.max(abs(self.z[layer]))
            self.a[layer] = self.phi(self.z[layer])
            i += 1
            layer += 1

        # replace last activation layer to be a softmax layer
        self.a[self.L-1] = self.softmax(self.z[self.L-1])
        return self.a[self.L-1]

    def softmax(self, z): # in this case z should be self.z(self.L-1)
        r = -np.max(z)
        exp_z = np.exp(z+r)
        q = np.sum(exp_z)
        return exp_z/q

    def loss(self, pred, y): # loss for a single (x,y)
        return -np.log(pred[np.argmax(y)])
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """

        # local gradients for the softmax layer
        iter = 0
        for p in self.a[self.L-1]:
            if (iter==np.argmax(y)):
                self.delta[self.L-1][iter]=p-1
            else:
                self.delta[self.L-1][iter]=p
            iter +=1

        # local gradients of the hidden layers
        layer = self.L-2
        while layer > 0:
            pd = self.phi_d(self.z[layer])
            for j in range(len(self.z[layer])):
                self.delta[layer][j] = pd[j]*(self.chain(self.delta[layer+1],self.w[layer+1],j))
            layer-=1

        layer=1

        # update derivates in terms of w and b
        while layer < self.L:
            self.dw[layer]= np.outer(self.delta[layer],self.a[layer-1])
            self.db[layer]=self.delta[layer]
            layer+=1

    # apply chain rule
    def chain(self,d,w,j):
        sum_over_k=0.0
        for k in range(len(d)):
            sum_over_k += d[k]*w[k][j]
        return sum_over_k

    # Return predicted image class for input x
    def predict(self, x):
        return np.argmax(self.forward(x))

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[self.L-1][j] 
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.00007,
            epochs=1000):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):
            #print(t)
            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                self.a     = [np.zeros(m) for m in [784,20,20,20,10]]
                self.z     = [np.zeros(m) for m in [784,20,20,20,10]]
                self.db    = [np.zeros(m) for m in [784,20,20,20,10]]
                self.dw    = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
                self.delta = [np.zeros(m) for m in [784,20,20,20,10]]
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    self.forward(x)
                    
                    # Compute gradients
                    self.backward(x,y)
                    
                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = self.w[l] - epsilon*self.dw[l]
                    self.b[l] = self.b[l] - epsilon*self.db[l]
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)


# Start training with default parameters.

def main():
    bp = BackPropagation()
    bp.sgd()

if __name__ == "__main__":
    main()
    

# -*- coding: utf-8 -*-
import numpy as np
from nn.utils import act_fct
from nn.utils import dact_fct



class NNet:
    """
    Class implementing a feed forward neural network.
    Fields:
        layers = a tuple containing numbers of neurons in each layer, starting from the input layer
        L = depth of the NN, eg, with depth L there are L matrices W: W[1], ...,W[L]
        act_hid = hidden activation function name, ie, a name of a activation function used by hidden layers
        act_out = output activation function name, ie, a name of activation function used by output layer
                     the names correspond to the names in utils.act_fct().
        W = dictionary containing the W matrices for each layer. The keys are arranged such that the matrices 
            stored in the dictionary corresponds to the notation form the lecture. Ie, W[1] is the matrix which
            describes the connections between the layer 0 and layer 1. The matrix stored at W[1] is a numpy array
            with dimensions (number of neurons in the layer 1) x (number of neurons in the layer 0)          
        b = dictionary containing the b vectors for each layer. The indexing corresponds to the indexing from
            the lecture. See above. Eg, dimensions of b[1] (number of neurons in the layer 1) x  1   
    """
	
	
	
	
    def __init__(self, layers, act_hidden, act_output):
        self.layers = layers
        self.L = len(layers) - 1
        self.act_hid = act_hidden
        self.act_out = act_output
        self.W, self.b = self.init_Wb()
    
    
    def init_Wb(self):
        """
        Initialize the matrices W[1],...,W[L] and the vectors b[1],...,b[L] with random number from gaussian
        distribution with 0-mean, and 0.1 variance. Note that W, b are dictionaries with integer keys.
        """



        W, b = {}, {}  #creation of empty dictionaries
        ###YOUR CODE HERE###
        for l in range(1, self.L+1):
            W[l] = np.random.normal(0, 0.1, (self.layers[l], self.layers[l-1]));
            b[l] = np.random.normal(0, 0.1, (self.layers[l], 1));


        return W, b


    def fp(self, x):
        """
        Forward propagation. Uses the current parameters W, b
        Inputs:
            x = np.array of size self.layers[0] x N. This means that this function
                performs the forward propagation for N input vectors (columns).
        Outputs:
            a = dictionary containing output of each layer of NN. Each dictionary stores N outputs
                for each of the inputs. Eg., a[1] should be np.array of size self.layers[1] x N
                The indexing corresponds to the indexing from the lecture. E.g. a[0]=x because a[0] 
                contains the N outputs of the input layer, which is the input x.
            z = dictionary containing input to each layer of NN. The indexing corresponds to the indexing
                from the lecture. E.g. z[1]=W[1].dot(a[0])+b[1].
        """
        a, z = {}, {}
        x = np.asarray(x,dtype=float)
        a[0] = x;

        ###YOUR CODE HERE###
        for l in range(1, self.L + 1):
            z[l] = (np.matmul(self.W[l],a[l-1]) + self.b[l])
            a[l] = act_fct(z[l], self.act_hid)
            if l == self.L:
                a[l] = act_fct(z[l], self.act_out)
        ###################
        return a,z
    
    def output(self, x):
        """
        Provides the output from the last layer of NN.
        """
        a,_ = self.fp(x)
        ###YOUR CODE HERE###
        a_out=a[self.L]
        ###################
        return a_out
    
    
    def bp(self, y, x, dCda_func, lbd=0):
        """
        Backpropagation. Uses the current parameters W, b
        Args:
            x = np.array of size self.layers[0] x N (contains N input vectors from the training set)
            y = np.array of size self.layers[L] x N (contains N output vectors from the training set)
        Returns:
            dW = dictionary corresponding to W, where each corresponding key contains a matrix of the 
                 same size, eg, W[i].shape = dW[i].shape for all i. It contains the partial derivatives
                 of the cost function with respect to each entry entry of W.
            db = dictionary corresponding to b, where each corresponding key contains a matrix of the 
                 same size, eg, b[i].shape = bW[i].shape for all i. It contains the partial derivatives
                 of the cost function with respect to each entry entry of b. 
            
        """

        dCdz={}
    ### YOUR CODE HERE ###
        a, z = self.fp(x)
        L = self.L
        dCdz[L] = dCda_func(y,a[L])*dact_fct(z[L], self.act_out)
        for l in range(L-1,0,-1):
            dCdz[l] = self.W[l+1].T.dot(dCdz[l+1]) * dact_fct(z[l], self.act_hid)

        db = {}
        for l in range(1, L + 1):
            db[l] = np.sum(dCdz[l], axis=1).reshape((-1, 1))

        dW = {}
        for l in range(1, L + 1):
            dW[l] = dCdz[l].dot(a[l - 1].T) + 2*lbd*self.W[l]

        ################
        return dW, db

    def gd_learn(self, iter_num, l_rate, y, x, dCda_func, lbd=0):
        """
        Performs gradient descent learning.
        """
        ### YOUR CODE HERE ###



        for i in range(iter_num):
            dW, db = self.bp(y, x, dCda_func,lbd)
            for l in range(1,self.L+1):
                self.W[l] = self.W[l] - l_rate * dW[l]
                self.b[l] = self.b[l] - l_rate * db[l]


        #####################
        return 0
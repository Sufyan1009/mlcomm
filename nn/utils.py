import numpy as np
import toolss.it as it
import matplotlib.pyplot as plt

def act_fct(x, type_fct):
    # type_fct=identity
    # type_fct=sigmoid
    # type_fct=tanh
    # type_fct=relu
    x=np.asarray(x,dtype=float)
    if type_fct == 'identity':
        y = x
    elif type_fct == "sigmoid":
        y = 1/(1 + np.exp(-x))
    elif type_fct == "tanh":
        y = np.tanh(x)
    elif type_fct == "relu":
        y = np.max(np.vstack((x, np.zeros(x.shape))), axis=0)
    else:
        raise ValueError("wrong option");

    return y


'''
def poly_extend_data1D(x):
    """
    Extend the provided input vector x, wtih subsequent powers of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN such that 1st row = x^0, 2nd row = x^1, ...
    """
    ### YOUR CODE HERE ###
    x = np.asarray(x);
    ones_vec=np.ones(np.size(x,axis=1)); #number of colums in x ( number of examples), and dimensions will be number of rows
    x_ext = np.vstack((ones_vec,x));
    return x_ext
'''


def poly_extend_data1D(x, p):
    """
    Extend the provided input vector x, wtih subsequent powers of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN such that 1st row = x^0, 2nd row = x^1, ...
    """
    ### YOUR CODE HERE ###
    x = np.asarray(x);
    ones_vec = np.ones(
        np.size(x, axis=1));  # number of colums in x ( number of examples), and dimensions will be number of rows
    x_e = np.vstack((ones_vec, x));
    for i in range(2, p + 1):
        x_e = np.vstack((x_e, x ** i))

    return x_e


def lir_cost(w, y, x):
    """
    Computes cost for linear regression with parameters w and data set x,y
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    cost = scalar
    """
    y_estimated = np.dot(w.T, x);
    l_cost = np.sum(0.5 * (np.square(y_estimated - y)));

    return l_cost


def lir_grad(w, y, x):
    """
    Returs gradient for linear regression with quadratic cost for parameter w and data set y, x.
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    gradT = np array of size Mx1
    """

    y_estimated = np.asarray(np.dot(w.T, x));
    # err or gradients
    err = y_estimated - y
    err = np.array(err)
    gradT = np.dot(x, err.T)
    return gradT


def gradient_descent(iter_num, l_rate, w_0, gradient_func,cost=None):
    """
    Performs gradient descent for iter_num iterations with learning rate l_rate from initial
    position w_0.
    w_0 = np array of size Mx1
    gradient_func(w) is a function which returns gradient for parameter w
    Output:
    w_opt = optimal parameters
    """
    c=np.zeros(iter_num);
    for x in range(0, iter_num):
        w_0 = w_0 - l_rate * gradient_func(w_0);

        c[x] = np.float32(cost(w_0))

    plt.plot(c)
    plt.show()

    w_opt = w_0
    return w_opt


def normalize_data(x):
    """
    Normalizes data. Should not normalize the first row (we assume it is the row of ones).
    x = np.array of size MxN
    Output:
    x_norm     = normalized np.array of size MxN
    norm_param = distionary with two keys "mean" and "var". Each key contains
    a np.array of size Mx1 with the mean and variance of each row of data array.
    For the first row,  set mean=0 and var=1
    """
    ### YOUR CODE HERE ###
    x = np.asarray(x);
    m = np.mean(x, axis=1).reshape((-1,1));
    v = np.var(x, axis=1).reshape((-1,1));
    m[0,] = 0;
    v[0,] = 1;
    x_norm = (x - m) / np.sqrt(v);
    dic = {'mean': m, 'var': v};
    return x_norm, dic


def sin_extend_data1D(x, p):
    """
    Extend the provided input vector x, wtih P subsequent sin harmonics of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN
    """
    ### YOUR CODE HERE ###
    x = np.asarray(x);
    ones_vec = np.ones(
        np.size(x, axis=1));  # number of colums in x ( number of examples), and dimensions will be number of rows
    har = np.sin(2 * np.pi * x / x.max())
    x_e = np.vstack((ones_vec, har));
    for i in range(2, p + 1):
        har = np.sin(2 * np.pi * i * x / x.max())
        x_e = np.vstack((x_e, har))

    return x_e


def poly_extend_data2D_(x, p=1):
    """
    Extend the provided input matrix x wtih all subsequent powers of terms of the input.
    x = np.array of size 2xN
    Output:
    x_e = np.array
    Eg. for p=3 and x of dimensions 2xN. x_e should be a matrix such that
    the 1st row is [1 1 .. 1], 2nd X[0,:], 3rd X[1,:], 4th X[0,:]**2,
    5th X[0,:]*X[1,:], 6th X[1,:]*2, 7th X[0,:]**3,  8th X[0,:]**2*X[1,:],
    and so on... till 10th row equal X[1,:]**3
    """
    ### YOUR CODE HERE ###
    x_tmp = [[x[0, :] ** (k - i) * x[1, :] ** i for i in range(k + 1)] for k in range(p + 1)]
    x_e = np.vstack(x_tmp)

    ### ######### ###
    return x_e


def poly_extend_data2D(x, p):
    """
    Extend the provided input matrix x wtih all subsequent powers of terms of the input.
    x = np.array of size 2xN
    Output:
    x_e = np.array
    Eg. for p=3 and x of dimensions 2xN. x_e should be a matrix such that
    the 1st row is [1 1 .. 1], 2nd X[0,:], 3rd X[1,:], 4th X[0,:]**2,
    5th X[0,:]*X[1,:], 6th X[1,:]*2, 7th X[0,:]**3,  8th X[0,:]**2*X[1,:],
    and so on... till 10th row equal X[1,:]**3
    """
    ### YOUR CODE HERE ###
    x_tmp = [[x[0, :] ** (k - i) * x[1, :] ** i for i in range(k + 1)] for k in range(p + 1)]
    x_e = np.vstack(x_tmp)
    #

    ### ######### ###
    return x_e


def lor_grad(w, y, x,lbd=0):
    """
    Returs gradient for logistic regression with the cross entropy cost function
    for parameter w and data set y, x.
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    gradT = np array of size Mx1
    """
    ## YOUR CODE HERE ###
    a = act_fct(np.dot(w.T, x), "sigmoid");
    err = a - y;
    err = np.array(err)
    gradT = np.dot(x, err.T)+lbd*w

    #####################
    return gradT


def lor_cost(w, y, x,lbd=0):
    """
    Computes cost for logistic regression with parameters w and data set x,y
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:  y_eq_0 = (y==0).nonzero()[1]

    cost = scalar
    """
    ## YOUR CODE HERE ###
    sigmoid_f = lambda x: 1 / (1 + np.exp(-x))
    y_eq_0 = (y == 0).nonzero()[1]
    y_eq_1 = (y == 1).nonzero()[1]
    cost = np.sum(-np.log(sigmoid_f(np.dot(w.T, x[:, y_eq_1])))) + np.sum(
        -np.log(1 - sigmoid_f(np.dot(w.T, x[:, y_eq_0]))))
    cost += lbd * np.sum(w ** 2)

    #####################
    return cost

def dact_fct(x, type_fct):
    """
    Implements derivatives of  activation functions to be used in Neural Networks. The
    Inputs:
        x = np.array of input values
        type_act =
             'identity' : for activation  y = f(x) = x
             'sigmoid': for activation y = f(x) = 1/(1+exp(-x))
             'tanh': for activation y = f(x) = tanh(x)
             'rect_lin_unit': for activation y = f(x) = max(x,0)
    Output:
       y = np.array containing f'(x)
    """
    # type_fct=identity
    # type_fct=sigmoid
    # type_fct=tanh
    # type_fct=rlu
    x = np.asarray(x, dtype=float)
    if type_fct == "identity":
        y = np.ones((x.size))
    elif type_fct == "sigmoid":
        y = np.divide(np.exp(-x),  ((1 + np.exp(-x))**2))
             #der of sigmoid is =(sig)*(1-sig)
    elif type_fct == "tanh":
        y = np.tanh(x)
        y = 1-(y**2);
    elif type_fct == "relu":
        #derivative of relu is 0 for x<=0 and 1 x>0
        x_z=(x<=0).nonzero()[0]
        x_nz=(x>0).nonzero()[0]
        x[x_z]=0;         x[x_nz]=1;
        y=x;


    else:
        raise ValueError("wrong option");

    ##################
    return y


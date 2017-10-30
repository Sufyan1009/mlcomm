
import numpy as np






def act_fct(x,type_fct):
    #type_fct=identity
    #type_fct=sigmoid
    # type_fct=tanh
    # type_fct=rlu
    x = np.asarray(x, dtype=float)
    if type_fct == "identity":
        y = x
    elif type_fct == "sigmoid":
        y = 1 / (1+ np.exp(-x))
    elif type_fct== "tanh":
        y = np.tanh(x)
    elif type_fct== "rlu":
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
    ones_vec = np.ones(np.size(x, axis=1));  # number of colums in x ( number of examples), and dimensions will be number of rows
    x_e = np.vstack((ones_vec, x));
    for i in range(2,p+1):
        x_e=np.vstack((x_e,x ** i))

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
    y_estimated=np.dot(w.T,x);
    l_cost = np.sum (0.5 * (np.square(y_estimated-y)));

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

    y_estimated = np.asarray(np.dot(w.T,x ));
    #err or gradients
    err=y_estimated-y
    err=np.array(err)
    gradT = np.dot(x,err.T)
    return gradT


def gradient_descent(iter_num, l_rate, w_0, gradient_func):
    """
    Performs gradient descent for iter_num iterations with learning rate l_rate from initial
    position w_0.
    w_0 = np array of size Mx1
    gradient_func(w) is a function which returns gradient for parameter w
    Output:
    w_opt = optimal parameters
    """
    for x in range(0,iter_num):
        w_0 = w_0 - l_rate * gradient_func(w_0);

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
    x=np.asarray(x);
    m=np.mean(x,axis=1);
    v=np.var(x,axis=1);
    m = m[np.newaxis, :].T;
    v = v[np.newaxis, :].T;
    m[0,]=0;
    v[0,]=1;
    x_norm=(x-m)/v;
    v=v**2; #we need to variance not std daviation
    dic={'mean':m,'var':v};
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
    ones_vec = np.ones(np.size(x, axis=1));  # number of colums in x ( number of examples), and dimensions will be number of rows
    har = np.sin(2 * np.pi * x / x.max())
    x_e = np.vstack((ones_vec, har));
    for i in range(2, p + 1):
        har=np.sin(2*np.pi*i*x/x.max())
        x_e = np.vstack((x_e, har))

    return x_e



#homework 4
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

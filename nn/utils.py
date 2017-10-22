
import numpy as np

def act_fct(x,type_fct):
    #type_fct=identity
    #type_fct=sigmoid
    # type_fct=tanh
    # type_fct=rlu

    if type_fct == "identity":
        return x
    elif type_fct== "sigmoid":
        return 1 / (1+ np.exp(-x))
    elif type_fct== "tanh":
        return np.tanh(x)
    elif type_fct== "rlu":
        return np.max([0,x])
    else:
        print ("wrong option")
        exit(1)


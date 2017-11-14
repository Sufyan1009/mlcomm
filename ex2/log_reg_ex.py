import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import nn.utils as nnu
from dataset2_logreg import DataSet

# get data
y_D, x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()

# extend x
lbd=0.1
pp=8
x_D = nnu.poly_extend_data2D(x_D,pp)

# random init of weights
### YOUR CODE HERE ###

#generating w of size Mx1 as X is of MxN
w = np.random.normal(0,1,(x_D.shape[0],1))

# normalization:
x_D, norm_param = nnu.normalize_data(x_D)


# plot and print cost
def extension_wrapper(x):
    return nnu.poly_extend_data2D(x,pp)


DataSet.plot_decision_boundary(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lor_cost(w, y_D, x_D,lbd))


# compute gradient and do gradient descent
def gradient_wrapper(w):
    return nnu.lor_grad(w, y_D, x_D,lbd)

def cost_wrapper(w):
    return nnu.lor_cost(w, y_D, x_D,lbd)



w = nnu.gradient_descent(10000, 0.1, w, gradient_wrapper, cost_wrapper)

# plot and print cost
DataSet.plot_decision_boundary(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lor_cost(w, y_D, x_D))

# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import nn.utils as nnu
from dataset2_linreg import DataSet

#get and plot the data
y_D,x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()


#extend x with ones:
#for dataset1 error<0.03,pp=3,numb_iter=1800,learn_rate=0.02
#for data set2, error=3.92 for ajeeb values
pp=4
#x_D = nnu.poly_extend_data1D(x_D,pp)
x_D = nnu.sin_extend_data1D(x_D,pp)

#random init of w:
#generating w of size Mx1 as X is of MxN
w=np.random.randn(np.size(x_D,axis=0),1)

#normalization:
x_D, norm_param = nnu.normalize_data(x_D)


#plot and compute cost
def extension_wrapper(x,p=pp):
    return nnu.sin_extend_data1D(x,p=pp)
#this also takes norm_param (mean and varience) to exactly plot the data as it was provided
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))


#compute gradient and do gradient descent
#function is passed in function as param now we only has to pass w
def gradient_wrapper(w):
    return nnu.lir_grad(w, y_D, x_D)
#by increaing number of iterations error decreased
w = nnu.gradient_descent(2000, 0.002, w, gradient_wrapper)


#plot and compute cost
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))

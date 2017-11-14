# -*- coding: UTF-8 -*-
import sys
sys.path.append('../')
import numpy as np
import numpy.testing as npt
import nn.nnet as nnn



def test_init():
    print('Testing initialization...')
    nn_layers = (1,2,3,1)
    nn = nnn.NNet(nn_layers, 'sigmoid', 'identity')        
    assert(nn.W[1].shape == (2,1))
    print('Size W[1] OK')
    assert(nn.W[2].shape == (3,2))
    print('Size W[2] OK')
    assert(nn.W[3].shape == (1,3))
    print('Size W[3] OK')
    assert(nn.b[1].shape == (2,1))
    print('Size b[1] OK')
    assert(nn.b[2].shape == (3,1))
    print('Size b[2] OK')
    assert(nn.b[3].shape == (1,1))
    print('Size b[3] OK')

def test_fp():
    print('Testing forward propagation...')
    nn_layers = (1,2,1)
    nn = nnn.NNet(nn_layers, 'sigmoid', 'identity')
    nn.W[1] = np.array((1.0, 2.0),dtype=float).reshape((2,1))
    nn.W[2] = np.array((-5.0/6.0, 1.0/3.0),dtype=float).reshape((1,2))
    nn.b[1] = np.array((1,0),dtype=float).reshape((-1,1))
    nn.b[2] = np.array((1.0/3.0),dtype=float).reshape((1,1))
    test_input = np.array((-3,-2,-1,0,1,2,3)).reshape((1,-1))
    a,z = nn.fp(test_input)
    at = {0: np.array([[-3, -2, -1,  0,  1,  2,  3]]),
    		1: np.array([[ 0.1192 ,  0.26894,  0.5    ,  0.73106,  0.8808 ,  0.95257,  0.98201],[ 0.00247,  0.01799,  0.1192 ,  0.5    ,  0.8808 ,  0.98201,  0.99753]]),
    		2: np.array([[ 0.23482,  0.11521, -0.0436 , -0.10922, -0.10707, -0.13314, -0.1525 ]])}
    zt ={ 1: np.array([[-2, -1,  0,  1,  2,  3,  4], [-6, -4, -2,  0,  2,  4,  6]]),
    		2: np.array([[ 0.23482,  0.11521, -0.0436 , -0.10922, -0.10707, -0.13314, -0.1525 ]])}   
    npt.assert_almost_equal(at[0],a[0],decimal=5)
    print('Output a[0] OK')
    npt.assert_almost_equal(zt[1], z[1], decimal=5)
    print('Output z[1] OK')
    npt.assert_almost_equal(at[1],a[1],decimal=5)
    print('Output a[1] OK')
    npt.assert_almost_equal(zt[2],z[2],decimal=5)
    print('Output z[2] OK')
    npt.assert_almost_equal(at[2],a[2],decimal=5)
    print('Output a[2] OK')

    
    
def test_bp():
    print('Testing back propagation...')
    nn_layers = (1,3,2,1)
    nn = nnn.NNet(nn_layers, 'sigmoid', 'sigmoid')
    nn.W[1] = np.array((1.0, 2.0, 0.7)).reshape((3,1))
    nn.W[2] = np.array((1./6,2./6,3./6,4./6,5./6,1.)).reshape((2,3))
    nn.W[3] = np.array((0.3,0.7)).reshape((1,2))
    nn.b[1] = np.array((1.,0.,0.8)).reshape((-1,1))
    nn.b[2] = np.array((1.,0.3)).reshape((-1,1))
    nn.b[3] = np.array(0.1).reshape((1,1))
    test_input = np.array((-3.,-2,-1,0,1,2,3)).reshape((1,-1))
    test_Output = np.array((1,1,1,0,0,0,0)).reshape((1,-1))
    dW,db = nn.bp(test_Output, test_input,  lambda y,a: (a-y)/a/(1-a))
    dWt = {1: np.array([[ 0.0374 ],[ 0.0111 ],[ 0.08616]]),
		2: np.array([[ 0.07735,  0.08299,  0.06904],[ 0.09869,  0.11735,  0.08177]]),
		3: np.array([[ 1.81627,  2.03337]])}
    dbt = {1: np.array([[-0.00354],[ 0.01491],[-0.00359]]),
		2: np.array([[ 0.05586],[ 0.02582]]), 
		3: np.array([[ 2.01176]])}
    npt.assert_almost_equal(dWt[1],dW[1],decimal=5)
    print('Output dW[1] OK')
    npt.assert_almost_equal(dWt[2],dW[2],decimal=5)
    print('Output dW[2] OK')
    npt.assert_almost_equal(dWt[3],dW[3],decimal=5)
    print('Output dW[3] OK')
    npt.assert_almost_equal(dbt[1],db[1],decimal=5)
    print('Output db[1] OK')
    npt.assert_almost_equal(dbt[2],db[2],decimal=5)
    print('Output db[2] OK')
    npt.assert_almost_equal(dbt[3],db[3],decimal=5)
    print('Output db[2] OK')

test_init()
test_fp()
test_bp()
import sys
sys.path.append('../../')
import unittest
import nn.utils as utils



class MLcommNNTests(unittest.TestCase):
    def test_act_fct(self):
        self.assertSequenceEqual(utils.act_fct([1, 2, 3, 4], 'identity').round(4).tolist(), [1,2,3,4])
        self.assertSequenceEqual(utils.act_fct([1, 2, 3, 4], 'sigmoid').round(4).tolist(), [0.7311, 0.8808, 0.9526, 0.9820])
        self.assertSequenceEqual(utils.act_fct([1, 2, 3, 4], 'tanh').round(4).tolist(), [0.7616, 0.9640, 0.9951, 0.9993])
        self.assertSequenceEqual(utils.act_fct([-5, 2, -3, 4], 'rlu').round(4).tolist(), [0, 2, 0, 4])
        self.assertRaises(ValueError, utils.act_fct, 'test_fun', 34)


if __name__ == '__main__':
    unittest.main()

from scipy import stats
import numpy as np


def discrete_entr(pX):
    hX = stats.entropy(pX);
    return hX;


import unittest


class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(discrete_entr((3, 3)), [0.69314718055994529])

    def test(self):
        self.assertEqual(discrete_entr((3)), [0.0])

if __name__ == '__main__':
    unittest.main()



def discrete_cross_entr(pX, pY):
    for ai in pX:
        if ai < 0:
            print("must be pos pdf")

    for ai in pY:
        if ai < 0:
            print("must be pos pdf")
            exit(1)

    if np.size(pX) != np.size(pY):
        exit(1)

    pX = pX / sum(pX);
    pY = pY / sum(pY);
    return np.sum(np.nan_to_num(-pX * np.log(pY)))



'''
 class MyTest(unittest.TestCase):
         def test(self):
            self.assertEqual(discrete_cross_entr((3, 3),(3,3)), [0.69314718055994529])

     def test(self):
         self.assertEqual(discrete_cross_entr((3,3)), [0.0])


     if __name__ == '__main__':
       unittest.main()
'''

def discrete_kl_dis(pX,pY):
    hX = stats.entropy(pX,pY);
    return hX;



"""
class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(discrete_kl_dis((3, 3)), [0.69314718055994529])

    def test(self):
        self.assertEqual(discrete_kl_dis((3)), [0.0])


if __name__ == '__main__':
    unittest.main()
"""

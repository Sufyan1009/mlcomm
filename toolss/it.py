from scipy import stats
import numpy as np



def discrete_entr(pX):
    """
       Calculates the entropy of the distribution pX, i.e.,
       sum(-pX*log2(pX))
    """

    pX=np.asarray(pX,dtype=float)

    if (abs(sum(pX)-1))> 1e-6:
        raise ValueError('sum of probs is not equal to 1')

    avec=pX < 0;
    if (sum(avec)>0):
        raise ValueError('neg probs are not allowed')

    pX=pX/sum(pX);

    bvec= pX > 0;
    hX=np.sum(-pX[bvec]*np.log2(pX[bvec]))

    #hX = stats.entropy(pX);



    return hX;



def discrete_cross_entr(pX, pY):

    pX = np.asarray(pX,dtype='float')
    pY = np.asarray(pY,dtype='float')

    if np.size(pX) != np.size(pY):
       raise ValueError('pX and pY must be of same size')


    if abs(sum(pX)-1) >1e-6:
        raise ValueError('sum of pX must be equal to 1')
    else:
        pX = pX / sum(pX);


    if abs(sum(pY)-1) >1e-6:
        raise ValueError('sum of pY must be equal to 1')
    else:
        pY = pY / sum(pY);

    avec = pX < 0;
    if (sum(avec) > 0):
        raise ValueError('neg prob pX are not allowed')

    bvec = pY < 0;
    if (sum(bvec) > 0):
        raise ValueError('neg prob pY are not allowed')

    avec = pX > 0;
    bvec = pY > 0;

    H= np.sum(-pX[avec] * np.log2(pY[bvec]));
    #return np.sum(np.nan_to_num(-pX * np.log(pY)))

    return H


def discrete_kl_div(pX,pY):

    kl = -discrete_entr(pX)+discrete_cross_entr(pX,pY);


    return kl;


